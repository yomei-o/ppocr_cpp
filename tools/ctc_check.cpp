// Does ctc_loss.hpp compute the gradient it claims? Checked against central differences, and
// against three properties the analytic form must satisfy whatever the numbers are.
//
//   sh tools/build_ctc_check.sh && ./ctc_check.exe
//
// The float32 pitfall from the sibling repo applies here too: perturbing one input at a time and
// comparing to a difference quotient is fine for a loss of order 1, which this is, but the step
// has to sit above the noise floor. 1e-3 on a log-prob is comfortably there and still small enough
// that the second-order term stays under the tolerance.
#include "../pure/ctc_loss.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

static std::vector<float> log_softmax(const std::vector<float>& z, int64_t T, int64_t C) {
  std::vector<float> out(z.size());
  for (int64_t t = 0; t < T; ++t) {
    const float* r = &z[(size_t)(t * C)];
    float m = r[0];
    for (int64_t c = 1; c < C; ++c) m = std::max(m, r[c]);
    double s = 0;
    for (int64_t c = 0; c < C; ++c) s += std::exp(r[c] - m);
    float ls = m + (float)std::log(s);
    for (int64_t c = 0; c < C; ++c) out[(size_t)(t * C + c)] = r[c] - ls;
  }
  return out;
}

int main() {
  const int64_t T = 12, C = 7;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> logits((size_t)(T * C));
  for (auto& v : logits) v = nd(rng);

  std::vector<int> target = {3, 3, 1, 5};      // a repeat, so the mandatory blank path matters
  std::vector<float> lp = log_softmax(logits, T, C);

  ctc::LossOut r = ctc::loss(lp.data(), T, C, target);
  if (!r.ok) { printf("FAIL: loss did not run\n"); return 1; }
  printf("loss = %.6f   (T=%lld C=%lld target len=%zu)\n", r.loss, (long long)T, (long long)C,
         target.size());

  // ---- property 1: a loss is -ln P, so P must be a probability ----
  double P = std::exp(-(double)r.loss);
  printf("P = %.6g  %s\n", P, (P > 0 && P <= 1.0 + 1e-6) ? "(in (0,1])" : "OUT OF RANGE");
  if (!(P > 0 && P <= 1.0 + 1e-6)) return 1;

  // ---- property 2: rows of the gradient wrt log-probs sum to -1 ----
  // Every timestep is occupied by exactly one path symbol, so the occupancies at t sum to 1 and
  // the gradient row sums to -1. This catches a whole class of indexing bugs that a single
  // finite-difference spot check would walk straight past.
  double worst_row = 0;
  for (int64_t t = 0; t < T; ++t) {
    double s = 0;
    for (int64_t c = 0; c < C; ++c) s += r.grad[(size_t)(t * C + c)];
    worst_row = std::max(worst_row, std::fabs(s + 1.0));
  }
  printf("max |row sum + 1| = %.3e  %s\n", worst_row, worst_row < 1e-4 ? "OK" : "FAIL");
  if (worst_row >= 1e-4) return 1;

  // ---- property 3: central differences on the log-probs ----
  // Perturbing a log-prob directly (not the logit) is what the returned gradient is against.
  const float h = 1e-3f;
  double worst = 0, worst_abs = 0, wnum = 0, wana = 0;
  long long wt = -1, wc = -1;
  int checked = 0;
  for (int64_t t = 0; t < T; t += 3) {
    for (int64_t c = 0; c < C; ++c) {
      std::vector<float> a = lp, b = lp;
      a[(size_t)(t * C + c)] += h;
      b[(size_t)(t * C + c)] -= h;
      ctc::LossOut ra = ctc::loss(a.data(), T, C, target);
      ctc::LossOut rb = ctc::loss(b.data(), T, C, target);
      double num = (ra.loss - rb.loss) / (2.0 * h);
      double ana = r.grad[(size_t)(t * C + c)];
      double den = std::max(1e-3, std::fabs(num) + std::fabs(ana));
      double rel = std::fabs(num - ana) / den;
      if (rel > worst) { worst = rel; wt = t; wc = c; wnum = num; wana = ana; }
      worst_abs = std::max(worst_abs, std::fabs(num - ana));
      ++checked;
    }
  }
  // Judged on the absolute difference, not the relative one. The worst *relative* entry is always
  // a cell whose true gradient is ~0 (a symbol that no alignment can occupy at that step): the
  // analytic value is exactly 0, the difference quotient is float noise, and their ratio is
  // meaningless. The printout below shows both so that claim is checkable rather than asserted.
  printf("central diff: %d entries, worst abs %.3e  %s\n"
         "   worst-relative entry: t=%lld c=%lld  numeric=%.3e  analytic=%.3e  (rel %.3e)\n",
         checked, worst_abs, worst_abs < 2e-3 ? "OK" : "FAIL", wt, wc, wnum, wana, worst);
  if (worst_abs >= 2e-3) return 1;

  // ---- property 4: a target that cannot fit must be refused, not silently wrong ----
  std::vector<int> too_long;
  for (int i = 0; i < 20; ++i) too_long.push_back(1 + (i % (int)(C - 1)));
  ctc::LossOut bad = ctc::loss(lp.data(), T, C, too_long);
  printf("target longer than T: ok=%s  %s\n", bad.ok ? "true" : "false", !bad.ok ? "OK" : "FAIL");
  if (bad.ok) return 1;

  printf("ctc_check: OK\n");
  return 0;
}
