// Backward for the ops the ONNX interpreter runs, so a graph can be fine-tuned.
//
// HOW THIS SITS NEXT TO THE FORWARD CODE. Nothing here reimplements a forward pass. Each wrapper
// calls the existing fast function — ew::relu, nd::matmul, conv2d_gen — and then, only when
// infer_only() is off, records parents and a backward closure. So:
//
//   * the inference path is untouched. onnx_run.hpp keeps calling into the same hoisted-pointer
//     loops ew.hpp was written for; the only addition is one predictable branch per graph node,
//     not per element.
//   * there is one dispatch, not two. A separate training interpreter would be 400 lines of
//     duplicated op decoding that drifts the first time either side is fixed.
//
// WHAT CARRIES A GRADIENT. Only the float value stream. ONNX graphs also move int64 shape data
// (Shape -> Slice -> Concat -> Reshape); the interpreter keeps those in a separate IVal store and
// they are constants as far as training is concerned. That is why Reshape's backward only needs to
// restore a shape and never touches its second input.
//
// THE CONVENTION IS autograd.hpp's, and it matters: the closure captures parents by shared_ptr and
// the output as a raw Node* — capturing the output by shared_ptr would make the node own a closure
// that owns the node, and nothing would ever be freed.
#pragma once
#include "autograd.hpp"
#include "ew.hpp"
#include "nd.hpp"
#include "nn_ops.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace gr {

inline int64_t numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Attach a tape entry. No-op under infer_only(), which is what keeps inference at its old speed.
//
// The grad buffer is (re)allocated here rather than trusted, because the forward helpers this file
// wraps deliberately do not have one:
//
//   * eight functions in nn_ops.hpp end in `return nograd(o)`, which clears grad and shrinks it —
//     the interpreter never wants it and on these graphs the allocate-and-zero measured as more
//     than the arithmetic;
//   * ew::reshape builds its node with make_shared<Node> and skips make_tensor entirely.
//
// Both are right for inference and both leave an empty vector that a backward closure would index
// straight past the end of. One check here covers every op instead of each wrapper remembering.
template <class F>
inline const Tensor& tape(const Tensor& out, std::vector<Tensor> parents, F&& bw) {
  if (infer_only()) return out;
  if (out->grad.size() != out->data.size()) out->grad.assign(out->data.size(), 0.f);
  out->requires_grad = true;
  out->parents = std::move(parents);
  out->backward_fn = std::forward<F>(bw);
  return out;
}

// ---------------------------------------------------------------------------------------------
// elementwise, same shape in and out
// ---------------------------------------------------------------------------------------------
// d/dx of each is written against the *output* where that is cheaper and exact:
//   relu'(x)        = [x > 0]                       (uses x)
//   sigmoid'(x)     = y(1-y)                        (uses y)
//   hardsigmoid'(x) = 1/6 on -3 < x < 3, else 0     (uses x)
//   hardswish'(x)   = x*hardsigmoid'(x) + hardsigmoid(x)
// The hard* kinks at exactly +-3 are measure zero; the choice below (derivative 0 at the kink)
// matches PyTorch.
inline Tensor relu(const Tensor& x) {
  Tensor o = ew::relu(x);
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) if (x->data[i] > 0.f) x->grad[i] += op->grad[i];
  });
}

inline Tensor sigmoid(const Tensor& x) {
  Tensor o = ew::sigmoid(x);
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) {
      const float y = op->data[i];
      x->grad[i] += op->grad[i] * y * (1.f - y);
    }
  });
}

inline Tensor hardsigmoid(const Tensor& x, float alpha, float beta) {
  Tensor o = ew::hardsigmoid(x, alpha, beta);
  Node* op = o.get();
  return tape(o, {x}, [x, op, alpha, beta] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) {
      const float v = alpha * x->data[i] + beta;
      if (v > 0.f && v < 1.f) x->grad[i] += op->grad[i] * alpha;
    }
  });
}

inline Tensor hardswish(const Tensor& x) {
  Tensor o = ew::hardswish(x);
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) {
      const float v = x->data[i];
      // y = x * clamp(x/6 + 1/2, 0, 1);  dy/dx = clamp(...) + x/6 inside the ramp
      const float h = v / 6.f + 0.5f;
      float d;
      if (h <= 0.f)      d = 0.f;
      else if (h >= 1.f) d = 1.f;
      else               d = h + v / 6.f;
      x->grad[i] += op->grad[i] * d;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// shape-only ops: the gradient is a permutation or a copy, never arithmetic
// ---------------------------------------------------------------------------------------------
// Reshape / Squeeze / Unsqueeze / Flatten all reduce to this: the buffer is unchanged, so the
// gradient is added straight back element for element.
inline Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape) {
  Tensor o = ew::reshape(x, shape);
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) x->grad[i] += op->grad[i];
  });
}

// Transpose. The gradient of a permutation is the inverse permutation, so this runs the forward
// again rather than hand-rolling an index walk: same code path, same tested strides, and no second
// place to get the ordering backwards. The cost is one temporary the size of the gradient.
inline Tensor transpose(const Tensor& x, const std::vector<int64_t>& perm) {
  Tensor o = nd::transpose(x, perm);
  Node* op = o.get();
  const std::vector<int64_t> os = o->shape;
  return tape(o, {x}, [x, op, perm, os] {
    std::vector<int64_t> inv(perm.size());
    for (size_t d = 0; d < perm.size(); ++d) inv[(size_t)perm[d]] = (int64_t)d;
    bool save = infer_only();
    infer_only() = true;                       // the temporary must not grow a tape of its own
    Tensor g = from_data(os, std::vector<float>(op->grad.begin(), op->grad.end()));
    Tensor gx = nd::transpose(g, inv);
    infer_only() = save;
    for (size_t i = 0; i < gx->data.size(); ++i) x->grad[i] += gx->data[i];
  });
}

// ---------------------------------------------------------------------------------------------
// broadcast binary ops
// ---------------------------------------------------------------------------------------------
// THE ONE THING THAT IS EASY TO GET WRONG. nd::binary broadcasts, so an input of shape [1,C,1,1]
// against [N,C,H,W] is *read* N*H*W times. Its gradient is therefore the SUM over those positions,
// not a copy — get this backwards and a bias-shaped parameter trains at N*H*W times the intended
// rate while everything still runs and slowly diverges.
//
// reduce_to() walks the output index space once and folds each contribution into the input cell it
// came from, using the same right-aligned broadcasting rule nd::binary uses forward: a dimension
// of 1 in the input maps every output coordinate to index 0.
inline void reduce_to(const std::vector<float>& src, const std::vector<int64_t>& oshape,
                      std::vector<float>& dst, const std::vector<int64_t>& ishape) {
  const size_t R = oshape.size();
  const size_t off = R - ishape.size();               // right-aligned
  std::vector<int64_t> ostr(R, 1), istr(R, 0);
  for (int64_t d = (int64_t)R - 2; d >= 0; --d) ostr[(size_t)d] = ostr[(size_t)d + 1] * oshape[(size_t)d + 1];
  {
    int64_t s = 1;
    for (int64_t d = (int64_t)ishape.size() - 1; d >= 0; --d) {
      istr[off + (size_t)d] = (ishape[(size_t)d] == 1) ? 0 : s;   // a 1-dim contributes no stride
      s *= ishape[(size_t)d];
    }
  }
  const int64_t n = (int64_t)src.size();
  std::vector<int64_t> idx(R, 0);
  for (int64_t lin = 0; lin < n; ++lin) {
    int64_t rem = lin, si = 0;
    for (size_t d = 0; d < R; ++d) {
      idx[d] = rem / ostr[d];
      rem %= ostr[d];
      si += idx[d] * istr[d];
    }
    dst[(size_t)si] += src[(size_t)lin];
  }
}

// da/db are the local partial derivatives evaluated per output element, given (a_val, b_val,
// out_val). Passing the output too means Div and Pow can reuse it instead of recomputing.
template <class DA, class DB>
inline Tensor binary(const Tensor& a, const Tensor& b, const Tensor& o, DA da, DB db) {
  Node* op = o.get();
  const std::vector<int64_t> os = o->shape, as = a->shape, bs = b->shape;
  return tape(o, {a, b}, [a, b, op, os, as, db, da, bs] {
    const size_t R = os.size();
    const size_t offa = R - as.size(), offb = R - bs.size();
    std::vector<int64_t> ostr(R, 1), astr(R, 0), bstr(R, 0);
    for (int64_t d = (int64_t)R - 2; d >= 0; --d) ostr[(size_t)d] = ostr[(size_t)d + 1] * os[(size_t)d + 1];
    { int64_t s = 1;
      for (int64_t d = (int64_t)as.size() - 1; d >= 0; --d) { astr[offa + (size_t)d] = as[(size_t)d] == 1 ? 0 : s; s *= as[(size_t)d]; } }
    { int64_t s = 1;
      for (int64_t d = (int64_t)bs.size() - 1; d >= 0; --d) { bstr[offb + (size_t)d] = bs[(size_t)d] == 1 ? 0 : s; s *= bs[(size_t)d]; } }
    const int64_t n = op->numel();
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t rem = lin, ai = 0, bi = 0;
      for (size_t d = 0; d < R; ++d) {
        int64_t k = rem / ostr[d];
        rem %= ostr[d];
        ai += k * astr[d];
        bi += k * bstr[d];
      }
      const float g = op->grad[lin], av = a->data[(size_t)ai], bv = b->data[(size_t)bi];
      const float ov = op->data[lin];
      a->grad[(size_t)ai] += g * da(av, bv, ov);
      b->grad[(size_t)bi] += g * db(av, bv, ov);
    }
  });
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  return binary(a, b, nd::add(a, b), [](float, float, float) { return 1.f; },
                                     [](float, float, float) { return 1.f; });
}
inline Tensor sub(const Tensor& a, const Tensor& b) {
  return binary(a, b, nd::sub(a, b), [](float, float, float) { return 1.f; },
                                     [](float, float, float) { return -1.f; });
}
inline Tensor mul(const Tensor& a, const Tensor& b) {
  return binary(a, b, nd::mul(a, b), [](float, float bv, float) { return bv; },
                                     [](float av, float, float) { return av; });
}
inline Tensor div(const Tensor& a, const Tensor& b) {
  return binary(a, b, nd::div(a, b), [](float, float bv, float) { return 1.f / bv; },
                                     [](float, float bv, float ov) { return -ov / bv; });
}
// d/da a^b = b*a^(b-1);  d/db a^b = a^b * ln a, which is only defined for a > 0. The graphs here
// use Pow with a constant exponent (it is how LayerNorm's sqrt and rsqrt come out of the exporter),
// so the base is a variance — positive — and the exponent side is a Constant that never asks for a
// gradient. Guarding the log keeps a stray negative base from producing NaN that would then spread
// through every parameter in one step.
inline Tensor pow(const Tensor& a, const Tensor& b) {
  return binary(a, b, nd::binary(a, b, [](float x, float y) { return std::pow(x, y); }),
                [](float av, float bv, float) { return bv * std::pow(av, bv - 1.f); },
                [](float av, float, float ov) { return av > 0.f ? ov * std::log(av) : 0.f; });
}

// log with a floor, for turning the graph's softmax output into CTC's input.
//
// d/dx log(x) = 1/x blows up exactly where the model is confident it is *not* looking at a symbol,
// and an 18385-wide distribution has thousands of such entries every frame. The floor bounds that
// derivative at 1/eps and, more usefully, makes the loss finite when a target character sits on a
// frame the model gives zero probability — which is the normal state of affairs at step 0 of a
// fine-tune, not an error.
inline Tensor log_clamped(const Tensor& x, float eps) {
  Tensor o = ew::unary(x, [eps](float v) { return std::log(v < eps ? eps : v); });
  Node* op = o.get();
  return tape(o, {x}, [x, op, eps] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) {
      const float v = x->data[i];
      x->grad[i] += op->grad[i] / (v < eps ? eps : v);
    }
  });
}

inline Tensor sqrt(const Tensor& x) {
  Tensor o = ew::unary(x, [](float v) { return std::sqrt(v); });
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t n = op->numel();
    for (int64_t i = 0; i < n; ++i) {
      const float y = op->data[i];
      if (y > 0.f) x->grad[i] += op->grad[i] * 0.5f / y;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// softmax and mean
// ---------------------------------------------------------------------------------------------
// softmax backward is the Jacobian-vector product y * (g - <g,y>), done per row along `axis`.
// Written against the output, so no exponentials are recomputed and nothing overflows.
inline Tensor softmax(const Tensor& x, int64_t axis) {
  Tensor o = nd::softmax(x, axis);
  Node* op = o.get();
  const std::vector<int64_t> os = o->shape;
  return tape(o, {x}, [x, op, os, axis] {
    const int64_t R = (int64_t)os.size();
    const int64_t ax = axis < 0 ? axis + R : axis;
    int64_t inner = 1;
    for (int64_t d = ax + 1; d < R; ++d) inner *= os[(size_t)d];
    const int64_t A = os[(size_t)ax];
    const int64_t outer = op->numel() / (A * inner);
    for (int64_t ob = 0; ob < outer; ++ob) {
      for (int64_t in = 0; in < inner; ++in) {
        const int64_t base = ob * A * inner + in;
        double dot = 0;
        for (int64_t k = 0; k < A; ++k) {
          const int64_t i = base + k * inner;
          dot += (double)op->grad[i] * op->data[i];
        }
        for (int64_t k = 0; k < A; ++k) {
          const int64_t i = base + k * inner;
          x->grad[i] += op->data[i] * (op->grad[i] - (float)dot);
        }
      }
    }
  });
}

// ReduceMean over `axes`: each output cell is the average of K inputs, so each of those inputs
// receives g/K. Implemented with reduce_to() in reverse — scatter rather than gather — by walking
// the input index space and mapping it down to the output cell it fed.
inline Tensor reduce_mean(const Tensor& x, const std::vector<int64_t>& axes, bool keepdims) {
  Tensor o = ::reduce_mean(x, axes, keepdims);
  Node* op = o.get();
  const std::vector<int64_t> xs = x->shape;
  return tape(o, {x}, [x, op, xs, axes] {
    const size_t R = xs.size();
    std::vector<char> red(R, 0);
    int64_t K = 1;
    for (int64_t a : axes) {
      const size_t d = (size_t)(a < 0 ? a + (int64_t)R : a);
      red[d] = 1;
      K *= xs[d];
    }
    // strides over the reduced shape, kept at full rank (keepdims or not, the element order of the
    // output is the same; only its shape vector differs)
    std::vector<int64_t> ostr(R, 0);
    int64_t s = 1;
    for (int64_t d = (int64_t)R - 1; d >= 0; --d) {
      if (!red[(size_t)d]) { ostr[(size_t)d] = s; s *= xs[(size_t)d]; }
    }
    std::vector<int64_t> xstr(R, 1);
    for (int64_t d = (int64_t)R - 2; d >= 0; --d) xstr[(size_t)d] = xstr[(size_t)d + 1] * xs[(size_t)d + 1];
    const int64_t n = x->numel();
    const float inv = 1.f / (float)K;
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t rem = lin, oi = 0;
      for (size_t d = 0; d < R; ++d) {
        const int64_t k = rem / xstr[d];
        rem %= xstr[d];
        oi += k * ostr[d];
      }
      x->grad[(size_t)lin] += op->grad[oi] * inv;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------------------------
// dA = dC B^T, dB = A^T dC, per batch. nd::matmul broadcasts a rank-2 operand across the batch of
// the other (batch_a==1 or batch_b==1), which is how the recogniser applies one weight matrix to
// every time step — so that operand's gradient must be summed over the batch, exactly like the
// broadcast binaries above. Writing the loops directly rather than materialising two transposes
// keeps that accumulation obvious.
inline Tensor matmul(const Tensor& a, const Tensor& b) {
  Tensor o = nd::matmul(a, b);
  Node* op = o.get();
  return tape(o, {a, b}, [a, b, op] {
    const size_t ra = a->shape.size(), rb = b->shape.size();
    const int64_t M = a->shape[ra - 2], K = a->shape[ra - 1], N = b->shape[rb - 1];
    int64_t ba = 1, bb = 1;
    for (size_t i = 0; i + 2 < ra; ++i) ba *= a->shape[i];
    for (size_t i = 0; i + 2 < rb; ++i) bb *= b->shape[i];
    const int64_t batch = ba > bb ? ba : bb;
    for (int64_t t = 0; t < batch; ++t) {
      const float* A = a->data.data() + (ba == 1 ? 0 : t) * M * K;
      const float* B = b->data.data() + (bb == 1 ? 0 : t) * K * N;
      float* dA = a->grad.data() + (ba == 1 ? 0 : t) * M * K;
      float* dB = b->grad.data() + (bb == 1 ? 0 : t) * K * N;
      const float* dO = op->grad.data() + t * M * N;
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
          float acc = 0.f;
          const float* Br = B + k * N;
          const float* Or = dO + i * N;
          for (int64_t j = 0; j < N; ++j) acc += Or[j] * Br[j];
          dA[i * K + k] += acc;                       // dA = dC B^T
        }
      }
      for (int64_t k = 0; k < K; ++k) {
        for (int64_t i = 0; i < M; ++i) {
          const float av = A[i * K + k];
          if (av == 0.f) continue;
          const float* Or = dO + i * N;
          float* Br = dB + k * N;
          for (int64_t j = 0; j < N; ++j) Br[j] += av * Or[j];   // dB = A^T dC
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------------------------
// slice and concat: the gradient is the same index map, run the other way
// ---------------------------------------------------------------------------------------------
inline Tensor slice(const Tensor& x, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& steps) {
  Tensor o = nd::slice(x, starts, ends, axes, steps);
  Node* op = o.get();
  const std::vector<int64_t> xs = x->shape, os = o->shape;
  return tape(o, {x}, [x, op, xs, os, starts, ends, axes, steps] {
    const size_t r = xs.size();
    std::vector<int64_t> ax = axes;
    if (ax.empty()) { ax.resize(starts.size()); for (size_t i = 0; i < ax.size(); ++i) ax[i] = (int64_t)i; }
    std::vector<int64_t> st = steps;
    if (st.empty()) st.assign(ax.size(), 1);
    std::vector<int64_t> beg(r, 0), stp(r, 1);
    for (size_t i = 0; i < ax.size(); ++i) {
      const int64_t a = nd::norm_axis(ax[i], r), dim = xs[(size_t)a];
      int64_t s = starts[i] < 0 ? starts[i] + dim : starts[i];
      s = std::max<int64_t>(0, std::min(s, dim));
      beg[(size_t)a] = s;
      stp[(size_t)a] = st[i];
    }
    const nd::Shape sx = nd::strides_of(xs), so = nd::strides_of(os);
    const int64_t n = op->numel();
    for (int64_t lin = 0; lin < n; ++lin) {
      int64_t rem = lin, src = 0;
      for (size_t d = 0; d < r; ++d) {
        const int64_t c = rem / so[d];
        rem -= c * so[d];
        src += (beg[d] + c * stp[d]) * sx[d];
      }
      x->grad[(size_t)src] += op->grad[lin];
    }
  });
}

inline Tensor concat(const std::vector<Tensor>& xs, int64_t axis) {
  Tensor o = nd::concat(xs, axis);
  Node* op = o.get();
  std::vector<std::vector<int64_t>> shapes;
  for (const auto& t : xs) shapes.push_back(t->shape);
  const std::vector<int64_t> os = o->shape;
  return tape(o, xs, [xs, op, shapes, os, axis] {
    const size_t r = os.size();
    const int64_t ax = nd::norm_axis(axis, r);
    int64_t inner = 1;
    for (size_t d = (size_t)ax + 1; d < r; ++d) inner *= os[d];
    int64_t outer = 1;
    for (int64_t d = 0; d < ax; ++d) outer *= os[(size_t)d];
    int64_t off = 0;
    for (size_t k = 0; k < xs.size(); ++k) {
      const int64_t A = shapes[k][(size_t)ax];
      for (int64_t ob = 0; ob < outer; ++ob) {
        const float* src = op->grad.data() + (ob * os[(size_t)ax] + off) * inner;
        float* dst = xs[k]->grad.data() + ob * A * inner;
        for (int64_t i = 0; i < A * inner; ++i) dst[i] += src[i];
      }
      off += A;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// pooling
// ---------------------------------------------------------------------------------------------
// GlobalAveragePool: [N,C,H,W] -> [N,C,1,1]. Every input in a plane gets the same g/(H*W).
inline Tensor gap(const Tensor& x) {
  Tensor o = ew::gap(x);
  Node* op = o.get();
  return tape(o, {x}, [x, op] {
    const int64_t N = x->shape[0], C = x->shape[1];
    const int64_t HW = x->numel() / (N * C);
    const float inv = 1.f / (float)HW;
    for (int64_t nc = 0; nc < N * C; ++nc) {
      const float g = op->grad[nc] * inv;
      float* dst = x->grad.data() + nc * HW;
      for (int64_t i = 0; i < HW; ++i) dst[i] += g;
    }
  });
}

// AveragePool. The divisor is NOT kh*kw in general: avgpool2d counts the cells it actually
// averaged, and whether a padded position counts is count_include_pad. So the backward recounts the
// window exactly the way the forward did rather than assuming a constant — with
// count_include_pad=false the edge windows divide by less, and a backward that used kh*kw there
// would quietly scale the border gradients down.
inline Tensor avgpool(const Tensor& x, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                      int64_t pt, int64_t pl, int64_t pb, int64_t pr,
                      bool count_include_pad, bool ceil_mode) {
  Tensor o = avgpool2d(x, kh, kw, sh, sw, pt, pl, pb, pr, count_include_pad, ceil_mode);
  Node* op = o.get();
  const std::vector<int64_t> xs = x->shape, os = o->shape;
  return tape(o, {x}, [x, op, xs, os, kh, kw, sh, sw, pt, pl, count_include_pad] {
    const int64_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    const int64_t OH = os[2], OW = os[3];
    for (int64_t nc = 0; nc < N * C; ++nc) {
      const float* g = op->grad.data() + nc * OH * OW;
      float* dx = x->grad.data() + nc * H * W;
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
          int64_t cnt = 0;
          for (int64_t r = 0; r < kh; ++r) {
            const int64_t ih = oh * sh + r - pt;
            for (int64_t t = 0; t < kw; ++t) {
              const int64_t iw = ow * sw + t - pl;
              const bool inside = (ih >= 0 && ih < H && iw >= 0 && iw < W);
              if (inside || count_include_pad) ++cnt;
            }
          }
          if (!cnt) continue;
          const float gv = g[oh * OW + ow] / (float)cnt;
          for (int64_t r = 0; r < kh; ++r) {
            const int64_t ih = oh * sh + r - pt;
            if (ih < 0 || ih >= H) continue;
            for (int64_t t = 0; t < kw; ++t) {
              const int64_t iw = ow * sw + t - pl;
              if (iw < 0 || iw >= W) continue;
              dx[ih * W + iw] += gv;
            }
          }
        }
    }
  });
}

// ---------------------------------------------------------------------------------------------
// batch norm — statistics FROZEN
// ---------------------------------------------------------------------------------------------
// ew::batchnorm is the inference form: y = gamma * (x - running_mean) / sqrt(running_var + eps) +
// beta. This backward keeps it that way. mean and var stay constants; gamma, beta and x get
// gradients.
//
// That is a decision, not a shortcut, and it is the right one here for three reasons:
//
//  1. It is what the graph contains. An exported ONNX model carries the running statistics, not the
//     machinery to recompute them, and PP-OCRv5's recogniser has six BatchNormalization nodes with
//     nothing else attached.
//  2. Fine-tuning happens at batch sizes of one to eight text lines. Batch statistics estimated
//     from that few samples are noisy enough to *undo* a pretrained backbone — freezing BN during
//     fine-tuning is standard practice, not a compromise.
//  3. Recomputing batch statistics would make the forward pass depend on which other samples are
//     in the batch, so a single-sample check could never reproduce a training step. Everything
//     here stays checkable one sample at a time.
//
// With mean and var fixed the layer is a per-channel affine map, so:
//     s = gamma / sqrt(var + eps)
//     dx     = g * s
//     dgamma = sum over N,H,W of g * (x - mean)/sqrt(var + eps)
//     dbeta  = sum over N,H,W of g
inline Tensor batchnorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                        const Tensor& mean, const Tensor& var, float eps) {
  Tensor o = ew::batchnorm(x, gamma, beta, mean, var, eps);
  Node* op = o.get();
  // mean and var are deliberately absent from the parent list: they are constants of this layer.
  return tape(o, {x, gamma, beta}, [x, gamma, beta, mean, var, op, eps] {
    const int64_t N = x->shape[0], C = x->shape[1];
    const int64_t HW = x->numel() / (N * C);
    for (int64_t c = 0; c < C; ++c) {
      const float rstd = 1.f / std::sqrt(var->data[(size_t)c] + eps);
      const float s = gamma->data[(size_t)c] * rstd;
      const float mu = mean->data[(size_t)c];
      double dg = 0, db = 0;
      for (int64_t n = 0; n < N; ++n) {
        const int64_t base = (n * C + c) * HW;
        for (int64_t i = 0; i < HW; ++i) {
          const float g = op->grad[(size_t)(base + i)];
          x->grad[(size_t)(base + i)] += g * s;
          dg += (double)g * (x->data[(size_t)(base + i)] - mu) * rstd;
          db += g;
        }
      }
      gamma->grad[(size_t)c] += (float)dg;
      beta->grad[(size_t)c] += (float)db;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// convolution
// ---------------------------------------------------------------------------------------------
// conv2d_gen is im2col + one GEMM per (sample, group):
//     out[Cout_g x P] = w[Cout_g x K] @ col[K x P],   K = Cin_g*kh*kw,  P = OH*OW
// so the three gradients are the three ways to read that product:
//     dW   [Cout_g x K] += dO [Cout_g x P] @ col^T [P x K]
//     dCol [K x P]       = w^T [K x Cout_g] @ dO [Cout_g x P]      then scattered back by col2im
//     dBias[c]          += sum over samples and positions of dO
//
// THE COLUMN BUFFER IS REBUILT, NOT STORED. Keeping every conv's col matrix alive between forward
// and backward would be the fast choice, but K*P floats per node across 38 convolutions is a large
// multiple of the model itself, and this has to fine-tune inside a browser as well as on a
// workstation. im2col is a copy, not arithmetic; redoing it costs far less than the GEMMs around
// it. If training speed ever becomes the complaint, this is the first thing to trade.
//
// The padded positions in col are zero, so they contribute nothing to dW and are simply skipped
// when scattering dCol — which is why the bounds test appears in the scatter and nowhere else.
inline Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t pb, int64_t pr,
                     int64_t groups) {
  Tensor o = conv2d_gen(x, w, bias, sh, sw, pt, pl, pb, pr, groups);
  Node* op = o.get();
  std::vector<Tensor> parents = {x, w};
  if (bias) parents.push_back(bias);
  return tape(o, parents, [x, w, bias, op, sh, sw, pt, pl, groups] {
    const int64_t N = x->shape[0], Cin = x->shape[1], H = x->shape[2], W = x->shape[3];
    const int64_t Cout = w->shape[0], kh = w->shape[2], kw = w->shape[3];
    const int64_t OH = op->shape[2], OW = op->shape[3];
    const int64_t Cin_g = Cin / groups, Cout_g = Cout / groups;
    const int64_t K = Cin_g * kh * kw, P = OH * OW;

    std::vector<float> col((size_t)K * P), dcol((size_t)K * P);
    std::vector<float> dWg_tmp((size_t)(Cout_g * K));
    for (int64_t n = 0; n < N; ++n)
      for (int64_t g = 0; g < groups; ++g) {
        const float* xg = x->data.data() + (n * Cin + g * Cin_g) * H * W;
        // im2col, identical to the forward
        for (int64_t c = 0; c < Cin_g; ++c)
          for (int64_t r = 0; r < kh; ++r)
            for (int64_t s = 0; s < kw; ++s) {
              float* crow = col.data() + (((c * kh + r) * kw + s) * P);
              for (int64_t oh = 0; oh < OH; ++oh) {
                const int64_t ih = oh * sh - pt + r;
                float* dst = crow + oh * OW;
                if (ih < 0 || ih >= H) { for (int64_t ow = 0; ow < OW; ++ow) dst[ow] = 0.f; continue; }
                const float* src = xg + (c * H + ih) * W;
                for (int64_t ow = 0; ow < OW; ++ow) {
                  const int64_t iw = ow * sw - pl + s;
                  dst[ow] = (iw < 0 || iw >= W) ? 0.f : src[iw];
                }
              }
            }

        const float* dO = op->grad.data() + (n * Cout + g * Cout_g) * P;
        const float* wg = w->data.data() + (g * Cout_g) * K;
        float* dWg = w->grad.data() + (g * Cout_g) * K;

        // Both of these go through the same device seam the forward GEMM uses, so conv backward
        // gets Eigen on the CPU and lands on cuBLAS under -DUSE_CUDA without another line of code.
        // backend.hpp already had exactly the two transposed forms this needs:
        //   gemm_nt(A,B,C,M,N,Kc)  =  A[M x Kc] * B[N x Kc]^T
        //   gemm_tn(A,B,C,M,N,Kc)  =  A[Kc x M]^T * B[Kc x N]
        //
        // dW accumulates across samples and groups, so beta=1 — writing it with beta=0 into a
        // scratch and adding afterwards would be the same arithmetic and one more buffer.
        bk::gemm_nt_hosted(dO, col.data(), dWg_tmp.data(), Cout_g, K, P, 0.f);
        for (int64_t i = 0; i < Cout_g * K; ++i) dWg[i] += dWg_tmp[(size_t)i];

        bk::gemm_tn_hosted(wg, dO, dcol.data(), K, P, Cout_g, 0.f);   // dcol = w^T dO

        float* dxg = x->grad.data() + (n * Cin + g * Cin_g) * H * W;
        for (int64_t c = 0; c < Cin_g; ++c)                    // col2im
          for (int64_t r = 0; r < kh; ++r)
            for (int64_t s = 0; s < kw; ++s) {
              const float* drow = dcol.data() + (((c * kh + r) * kw + s) * P);
              for (int64_t oh = 0; oh < OH; ++oh) {
                const int64_t ih = oh * sh - pt + r;
                if (ih < 0 || ih >= H) continue;
                const float* src = drow + oh * OW;
                float* dst = dxg + (c * H + ih) * W;
                for (int64_t ow = 0; ow < OW; ++ow) {
                  const int64_t iw = ow * sw - pl + s;
                  if (iw < 0 || iw >= W) continue;
                  dst[iw] += src[ow];
                }
              }
            }
      }

    if (bias) {
      for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < Cout; ++c) {
          const float* g = op->grad.data() + (n * Cout + c) * P;
          double acc = 0;
          for (int64_t i = 0; i < P; ++i) acc += g[i];
          bias->grad[(size_t)c] += (float)acc;
        }
    }
  });
}

// Depthwise: groups == Cin == Cout, one kh*kw kernel per channel, w is [C,1,kh,kw]. The forward
// skips im2col entirely because a GEMM with M=1 is all overhead, and the backward does the same —
// the three gradients fall out of a single loop nest over the output.
inline Tensor dwconv2d(const Tensor& x, const Tensor& w, const Tensor& bias,
                       int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t pb, int64_t pr) {
  Tensor o = dwconv2d_gen(x, w, bias, sh, sw, pt, pl, pb, pr);
  Node* op = o.get();
  std::vector<Tensor> parents = {x, w};
  if (bias) parents.push_back(bias);
  return tape(o, parents, [x, w, bias, op, sh, sw, pt, pl] {
    const int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    const int64_t kh = w->shape[2], kw = w->shape[3];
    const int64_t OH = op->shape[2], OW = op->shape[3];
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c) {
        const float* g = op->grad.data() + ((n * C + c) * OH) * OW;
        const float* xc = x->data.data() + ((n * C + c) * H) * W;
        float* dxc = x->grad.data() + ((n * C + c) * H) * W;
        const float* wc = w->data.data() + c * kh * kw;
        float* dwc = w->grad.data() + c * kh * kw;
        double db = 0;
        for (int64_t oh = 0; oh < OH; ++oh)
          for (int64_t ow = 0; ow < OW; ++ow) {
            const float gv = g[oh * OW + ow];
            if (gv == 0.f) continue;
            db += gv;
            for (int64_t r = 0; r < kh; ++r) {
              const int64_t ih = oh * sh - pt + r;
              if (ih < 0 || ih >= H) continue;
              for (int64_t s = 0; s < kw; ++s) {
                const int64_t iw = ow * sw - pl + s;
                if (iw < 0 || iw >= W) continue;
                dwc[r * kw + s] += gv * xc[ih * W + iw];
                dxc[ih * W + iw] += gv * wc[r * kw + s];
              }
            }
          }
        if (bias) bias->grad[(size_t)c] += (float)db;
      }
  });
}

}  // namespace gr
