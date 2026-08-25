// CTC loss (forward-backward), for fine-tuning the recogniser.
//
// WHAT THIS TAKES. Log-probabilities, [T, C], not logits. The PP-OCR graph already ends in Softmax
// (see ctc.hpp — greedy() reads the output as probabilities), so the training path applies Log to
// that output and hands the result here. Everything downstream is then ordinary autograd: this
// returns d(loss)/d(log_probs), Log's backward turns that into d/d(probs), and Softmax's backward
// turns that into d/d(logits). Splitting it that way costs one extra elementwise pass and buys not
// having to find and cut the head out of a 746-node graph.
//
// WHY THE GRADIENT IS THIS SIMPLE. With y = probs, x = ln y, and
//     P = sum over all alignments,  loss = -ln P,
// the standard forward-backward identity is
//     sum_{s : l'_s = c} alpha[t][s] * beta[t][s]  ==  y[t][c] * dP/dy[t][c].
// So
//     dloss/dx[t][c] = (dloss/dy) * (dy/dx) = (-1/P)(dP/dy) * y = -(1/P) * sum_{s:l'_s=c} a*b.
// The y cancels. Against logits you would get the familiar `y - occupancy` form; against
// log-probs the y term is simply not there, and no division by a probability ever happens.
//
// EVERYTHING IS IN LOG SPACE. alpha and beta underflow float and even double quickly — a 40-step
// alignment multiplies 40 probabilities — so they are kept as logs and combined with logsumexp.
// The cost is a few exp/log per cell; the alternative is scaling factors per timestep, which is
// more code and more places to be subtly wrong.
//
// LABEL LAYOUT is ctc.hpp's, and it is not negotiable: index 0 is the blank, 1..N are the
// dictionary lines, N+1 is a literal space. Targets given to loss() are those indices, so a target
// may never contain 0.
#pragma once
#include "autograd.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace ctc {

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

inline float logaddexp(float a, float b) {
  if (a == NEG_INF) return b;
  if (b == NEG_INF) return a;
  float hi = a > b ? a : b, lo = a > b ? b : a;
  return hi + std::log1p(std::exp(lo - hi));
}

struct LossOut {
  float loss = 0.f;             // -ln P for this sample
  std::vector<float> grad;      // d(loss)/d(log_probs), [T*C], row-major
  bool ok = false;              // false when the target cannot fit in T steps
};

// log_probs: [T, C] row-major, natural log. target: label indices in 1..C-1, no blanks.
//
// The extended label l' interleaves blanks: b l_0 b l_1 b ... l_{L-1} b, length S = 2L+1. A path
// may skip a blank only between two *different* labels; between equal labels the blank is
// mandatory, which is the whole reason CTC can emit "aa".
inline LossOut loss(const float* log_probs, int64_t T, int64_t C,
                    const std::vector<int>& target) {
  LossOut out;
  const int64_t L = (int64_t)target.size();
  const int64_t S = 2 * L + 1;

  for (int v : target) {
    if (v <= 0 || v >= (int)C) {
      printf("ctc::loss: target id %d out of range 1..%lld\n", v, (long long)C - 1);
      return out;
    }
  }
  // Shortest legal path is L labels plus one blank between every equal-neighbour pair.
  int64_t need = L;
  for (int64_t i = 1; i < L; ++i) if (target[i] == target[i - 1]) ++need;
  if (T < need) return out;                       // caller skips this sample; ok stays false

  auto lp = [&](int64_t t, int64_t c) { return log_probs[t * C + c]; };
  auto lab = [&](int64_t s) -> int { return (s % 2 == 0) ? 0 : target[s / 2]; };

  std::vector<float> a((size_t)(T * S), NEG_INF), b((size_t)(T * S), NEG_INF);

  // forward
  a[0] = lp(0, 0);
  if (S > 1) a[1] = lp(0, lab(1));
  for (int64_t t = 1; t < T; ++t) {
    const float* ap = &a[(t - 1) * S];
    float* cur = &a[t * S];
    // A cell is unreachable if even the fastest path cannot arrive by t; skipping those is not an
    // optimisation, it keeps -inf out of logaddexp chains where it would be harmless but slow.
    int64_t smin = std::max<int64_t>(0, S - 2 * (T - t));
    for (int64_t s = smin; s < S; ++s) {
      float v = ap[s];
      if (s >= 1) v = logaddexp(v, ap[s - 1]);
      if (s >= 2 && lab(s) != 0 && lab(s) != lab(s - 2)) v = logaddexp(v, ap[s - 2]);
      cur[s] = v == NEG_INF ? NEG_INF : v + lp(t, lab(s));
    }
  }

  float logP = logaddexp(a[(T - 1) * S + S - 1], S >= 2 ? a[(T - 1) * S + S - 2] : NEG_INF);
  if (logP == NEG_INF) return out;                // no alignment survived

  // backward
  b[(T - 1) * S + S - 1] = 0.f;
  if (S >= 2) b[(T - 1) * S + S - 2] = 0.f;
  for (int64_t t = T - 2; t >= 0; --t) {
    const float* bp = &b[(t + 1) * S];
    float* cur = &b[t * S];
    for (int64_t s = 0; s < S; ++s) {
      float v = NEG_INF;
      auto take = [&](int64_t s2) {
        if (s2 < S && bp[s2] != NEG_INF) v = logaddexp(v, bp[s2] + lp(t + 1, lab(s2)));
      };
      take(s);
      take(s + 1);
      if (s + 2 < S && lab(s + 2) != 0 && lab(s + 2) != lab(s)) take(s + 2);
      cur[s] = v;
    }
  }

  // dloss/dlog_probs[t][c] = -exp( logsumexp_{s: l'_s = c}( a[t][s] + b[t][s] ) - logP )
  out.grad.assign((size_t)(T * C), 0.f);
  std::vector<float> acc((size_t)C);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < C; ++c) acc[(size_t)c] = NEG_INF;
    const float* ap = &a[t * S];
    const float* bp = &b[t * S];
    for (int64_t s = 0; s < S; ++s) {
      if (ap[s] == NEG_INF || bp[s] == NEG_INF) continue;
      int c = lab(s);
      acc[(size_t)c] = logaddexp(acc[(size_t)c], ap[s] + bp[s]);
    }
    float* g = &out.grad[(size_t)(t * C)];
    for (int64_t c = 0; c < C; ++c)
      if (acc[(size_t)c] != NEG_INF) g[c] = -std::exp(acc[(size_t)c] - logP);
  }

  out.loss = -logP;
  out.ok = true;
  return out;
}

// Tensor front end: log_probs is [1, T, C] or [T, C] and must carry a tape (infer_only() off).
//
// Returns a SCALAR loss node, not a number with the gradient poked into log_probs->grad. That is
// what autograd.hpp's backward() expects — it asserts the root is scalar and, before running any
// closure, zeroes the grad of every node it can reach. A gradient seeded ahead of that call would
// be erased by it. Wrapping the result as a node also means free_graph() tears this down like any
// other op, and a caller can scale or sum losses before calling backward.
//
// Returns nullptr when the target cannot fit in T steps; the caller skips that sample.
inline Tensor loss_node(const Tensor& log_probs, const std::vector<int>& target) {
  int64_t T, C;
  const auto& sh = log_probs->shape;
  if (sh.size() == 3 && sh[0] == 1) { T = sh[1]; C = sh[2]; }
  else if (sh.size() == 2)          { T = sh[0]; C = sh[1]; }
  else {
    printf("ctc::loss_node: expected [1,T,C] or [T,C]\n");
    return nullptr;
  }
  if (log_probs->grad.size() != (size_t)(T * C)) {
    printf("ctc::loss_node: log_probs has no grad buffer — infer_only() is still on\n");
    return nullptr;
  }
  LossOut r = loss(log_probs->data.data(), T, C, target);
  if (!r.ok) return nullptr;

  Tensor L = make_tensor({1}, true);
  L->data[0] = r.loss;
  L->parents = {log_probs};
  Node* op = L.get();
  std::vector<float> g = std::move(r.grad);
  // The chain rule factor is L->grad[0]: normally 1, but a caller may weight this sample.
  L->backward_fn = [log_probs, op, g] {
    const float s = op->grad[0];
    for (size_t i = 0; i < g.size(); ++i) log_probs->grad[i] += s * g[i];
  };
  return L;
}

}  // namespace ctc
