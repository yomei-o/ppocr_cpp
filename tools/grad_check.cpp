// Numerical gradient check for pure/onnx_grad.hpp.
//
//   g++ -std=c++17 -O1 -o grad_check.exe tools/grad_check.cpp && ./grad_check.exe
//
// Every op gets the same treatment: build a random input, run the op, reduce the output to a
// scalar with a fixed random projection, and compare backward()'s answer to central differences on
// each input element.
//
// WHY A RANDOM PROJECTION AND NOT sum(). Reducing with sum() sends the same gradient (1) into
// every output element, so an op that scrambles its output ordering — transpose with the wrong
// permutation, a reshape that should have been a copy — still passes. A different weight per
// output element makes the ordering observable. This is the same trap the sibling repo documents
// for whole-model checks, one level down.
//
// The step size is 1e-3 in float32. Smaller drowns in the ~1e-7 relative resolution of the loss;
// larger lets the second-order term show. Judged on absolute difference, because entries whose
// true gradient is zero produce a meaningless ratio (see tools/ctc_check.cpp).
#include "../pure/onnx_grad.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <functional>

static std::mt19937 g_rng(20260825);

static Tensor rand_tensor(std::vector<int64_t> shape, float scale = 1.f) {
  std::normal_distribution<float> nd(0.f, scale);
  Tensor t = make_tensor(shape, true);
  for (auto& v : t->data) v = nd(g_rng);
  return t;
}

// A fixed projection vector per output size, so the "loss" is reproducible across the perturbed
// runs of the same check.
static std::vector<float> projection(int64_t n, uint32_t seed) {
  std::mt19937 r(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> w((size_t)n);
  for (auto& v : w) v = nd(r);
  return w;
}

static int failures = 0;

// f maps the input tensor to an output tensor, building a tape.
static void check(const std::string& name, const Tensor& x,
                  const std::function<Tensor(const Tensor&)>& f) {
  printf("  %-16s ...", name.c_str()); fflush(stdout);
  infer_only() = false;
  Tensor y = f(x);
  std::vector<float> w = projection(y->numel(), 777);

  // scalar loss = <w, y>, taped so backward() has a scalar root
  Tensor L = make_tensor({1}, true);
  double s = 0;
  for (int64_t i = 0; i < y->numel(); ++i) s += (double)w[(size_t)i] * y->data[i];
  L->data[0] = (float)s;
  L->parents = {y};
  Node* lp = L.get();
  L->backward_fn = [y, lp, w] {
    for (int64_t i = 0; i < y->numel(); ++i) y->grad[i] += lp->grad[0] * w[(size_t)i];
  };
  backward(L);

  std::vector<float> ana(x->grad.begin(), x->grad.end());

  const float h = 1e-3f;
  double worst = 0;
  int64_t wi = -1;
  double wn = 0, wa = 0;
  for (int64_t i = 0; i < x->numel(); ++i) {
    const float keep = x->data[i];
    auto eval = [&](float v) {
      x->data[i] = v;
      infer_only() = true;                    // forward only; no tape for the probe
      Tensor yy = f(x);
      double acc = 0;
      for (int64_t k = 0; k < yy->numel(); ++k) acc += (double)w[(size_t)k] * yy->data[k];
      infer_only() = false;
      return acc;
    };
    double num = (eval(keep + h) - eval(keep - h)) / (2.0 * h);
    x->data[i] = keep;
    double d = std::fabs(num - ana[(size_t)i]);
    if (d > worst) { worst = d; wi = i; wn = num; wa = ana[(size_t)i]; }
  }
  const bool ok = worst < 3e-3;
  printf("  %-16s n=%-5lld worst abs %.3e  %s", name.c_str(), (long long)x->numel(), worst,
         ok ? "OK\n" : "FAIL\n");
  if (!ok) {
    printf("      at i=%lld numeric=%.6f analytic=%.6f\n", (long long)wi, wn, wa);
    ++failures;
  }
}

int main() {
  printf("onnx_grad backward vs central differences\n");

  check("relu", rand_tensor({2, 3, 4}), [](const Tensor& x) { return gr::relu(x); });
  check("sigmoid", rand_tensor({2, 3, 4}), [](const Tensor& x) { return gr::sigmoid(x); });
  check("hardswish", rand_tensor({2, 3, 4}, 2.f), [](const Tensor& x) { return gr::hardswish(x); });
  check("hardsigmoid", rand_tensor({2, 3, 4}, 2.f),
        [](const Tensor& x) { return gr::hardsigmoid(x, 1.f / 6.f, 0.5f); });
  check("reshape", rand_tensor({2, 3, 4}),
        [](const Tensor& x) { return gr::reshape(x, {6, 4}); });
  check("transpose", rand_tensor({2, 3, 4}),
        [](const Tensor& x) { return gr::transpose(x, {2, 0, 1}); });
  check("transpose2", rand_tensor({3, 5}),
        [](const Tensor& x) { return gr::transpose(x, {1, 0}); });

  // Broadcasting, both ways round. The stretched operand is the interesting one: its gradient has
  // to be the SUM over every position that read it, and checking only the full-size side would
  // never notice.
  {
    Tensor b = rand_tensor({1, 3, 1, 1});
    check("add_bcast", rand_tensor({2, 3, 2, 2}), [b](const Tensor& x) { return gr::add(x, b); });
    check("sub_bcast", rand_tensor({2, 3, 2, 2}), [b](const Tensor& x) { return gr::sub(x, b); });
    check("mul_bcast", rand_tensor({2, 3, 2, 2}), [b](const Tensor& x) { return gr::mul(x, b); });
    Tensor a = rand_tensor({2, 3, 2, 2});
    check("mul_bcast_rhs", rand_tensor({1, 3, 1, 1}),
          [a](const Tensor& x) { return gr::mul(a, x); });
    check("add_bcast_rhs", rand_tensor({1, 3, 1, 1}),
          [a](const Tensor& x) { return gr::add(a, x); });
  }
  {
    Tensor d = rand_tensor({2, 3});
    for (auto& v : d->data) v = std::fabs(v) + 0.7f;        // keep the divisor off zero
    check("div", rand_tensor({2, 3}), [d](const Tensor& x) { return gr::div(x, d); });
  }
  {
    Tensor pos = rand_tensor({2, 3});
    for (auto& v : pos->data) v = std::fabs(v) + 0.5f;
    check("sqrt", pos, [](const Tensor& x) { return gr::sqrt(x); });
  }
  {
    Tensor e = make_tensor({1}, false);
    e->data[0] = 1.5f;
    Tensor base = rand_tensor({2, 3});
    for (auto& v : base->data) v = std::fabs(v) + 0.5f;     // a^b needs a > 0 to be smooth
    check("pow", base, [e](const Tensor& x) { return gr::pow(x, e); });
  }
  check("softmax_last", rand_tensor({2, 5}), [](const Tensor& x) { return gr::softmax(x, -1); });
  check("softmax_mid", rand_tensor({2, 4, 3}), [](const Tensor& x) { return gr::softmax(x, 1); });
  check("reduce_mean", rand_tensor({2, 3, 4}),
        [](const Tensor& x) { return gr::reduce_mean(x, {2}, true); });
  check("reduce_mean_2ax", rand_tensor({2, 3, 4}),
        [](const Tensor& x) { return gr::reduce_mean(x, {1, 2}, false); });
  // Empty axes: ONNX reads that as "reduce every axis". Both cases above NAME their axes, which is
  // why the backward could divide by 1 instead of by the element count and stay hidden -- the
  // analytic gradient came out exactly 24x the numeric one here.
  check("reduce_mean_allax", rand_tensor({2, 3, 4}),
        [](const Tensor& x) { return gr::reduce_mean(x, {}, false); });

  // matmul, including the case the recogniser actually uses: one weight matrix applied to every
  // time step, so the weight's gradient is a sum over the batch.
  {
    Tensor w = rand_tensor({4, 5});
    check("matmul", rand_tensor({3, 4}), [w](const Tensor& x) { return gr::matmul(x, w); });
    Tensor bx = rand_tensor({2, 3, 4});
    check("matmul_bcastW", rand_tensor({4, 5}),
          [bx](const Tensor& x) { return gr::matmul(bx, x); });
    Tensor bw = rand_tensor({2, 4, 5});
    check("matmul_batched", rand_tensor({2, 3, 4}),
          [bw](const Tensor& x) { return gr::matmul(x, bw); });
  }
  check("slice", rand_tensor({2, 6, 3}),
        [](const Tensor& x) { return gr::slice(x, {1}, {5}, {1}, {2}); });
  {
    Tensor other = rand_tensor({2, 2, 3});
    check("concat", rand_tensor({2, 4, 3}),
          [other](const Tensor& x) { return gr::concat({x, other}, 1); });
    check("concat_rhs", rand_tensor({2, 2, 3}),
          [other](const Tensor& x) { return gr::concat({other, x}, 1); });
  }
  check("gap", rand_tensor({2, 3, 4, 5}), [](const Tensor& x) { return gr::gap(x); });
  check("avgpool_incl", rand_tensor({1, 2, 5, 5}),
        [](const Tensor& x) { return gr::avgpool(x, 3, 3, 2, 2, 1, 1, 1, 1, true, false); });
  // count_include_pad=false makes the edge windows divide by fewer cells; a backward that assumed
  // kh*kw would be wrong exactly there and nowhere else.
  check("avgpool_excl", rand_tensor({1, 2, 5, 5}),
        [](const Tensor& x) { return gr::avgpool(x, 3, 3, 2, 2, 1, 1, 1, 1, false, false); });
  {
    Tensor gm = rand_tensor({3}), bt = rand_tensor({3});
    Tensor mu = rand_tensor({3});
    Tensor va = rand_tensor({3});
    for (auto& v : va->data) v = std::fabs(v) + 0.5f;
    check("batchnorm_x", rand_tensor({2, 3, 2, 2}),
          [gm, bt, mu, va](const Tensor& x) { return gr::batchnorm(x, gm, bt, mu, va, 1e-5f); });
    Tensor xin = rand_tensor({2, 3, 2, 2});
    check("batchnorm_gamma", rand_tensor({3}),
          [xin, bt, mu, va](const Tensor& g) { return gr::batchnorm(xin, g, bt, mu, va, 1e-5f); });
    check("batchnorm_beta", rand_tensor({3}),
          [xin, gm, mu, va](const Tensor& b) { return gr::batchnorm(xin, gm, b, mu, va, 1e-5f); });
  }

  // Convolution, checked from all three sides. The stride/pad combinations are the ones the
  // recogniser actually uses: (2,1) and (1,2) to squeeze height while keeping the time axis, and a
  // 1x3 kernel padded (0,1,0,1) — the asymmetric cases where an index slip still produces plausible
  // numbers.
  {
    Tensor wt = rand_tensor({4, 2, 3, 3}, 0.3f), bs = rand_tensor({4}, 0.3f);
    check("conv_x", rand_tensor({1, 2, 5, 6}),
          [wt, bs](const Tensor& x) { return gr::conv2d(x, wt, bs, 1, 1, 1, 1, 1, 1, 1); });
    Tensor xi = rand_tensor({1, 2, 5, 6});
    check("conv_w", rand_tensor({4, 2, 3, 3}, 0.3f),
          [xi, bs](const Tensor& w) { return gr::conv2d(xi, w, bs, 1, 1, 1, 1, 1, 1, 1); });
    check("conv_b", rand_tensor({4}, 0.3f),
          [xi, wt](const Tensor& b) { return gr::conv2d(xi, wt, b, 1, 1, 1, 1, 1, 1, 1); });
    check("conv_stride21", rand_tensor({1, 2, 6, 6}),
          [wt, bs](const Tensor& x) { return gr::conv2d(x, wt, bs, 2, 1, 1, 1, 1, 1, 1); });
    Tensor w13 = rand_tensor({4, 2, 1, 3}, 0.3f);
    check("conv_1x3pad", rand_tensor({1, 2, 5, 6}),
          [w13, bs](const Tensor& x) { return gr::conv2d(x, w13, bs, 1, 1, 0, 1, 0, 1, 1); });
    // grouped: 2 groups of 2 in, 2 out each
    Tensor wg = rand_tensor({4, 1, 3, 3}, 0.3f);
    check("conv_groups2", rand_tensor({1, 2, 5, 6}),
          [wg, bs](const Tensor& x) { return gr::conv2d(x, wg, bs, 1, 1, 1, 1, 1, 1, 2); });
  }
  {
    Tensor wd = rand_tensor({3, 1, 3, 3}, 0.3f), bd = rand_tensor({3}, 0.3f);
    check("dwconv_x", rand_tensor({1, 3, 5, 6}),
          [wd, bd](const Tensor& x) { return gr::dwconv2d(x, wd, bd, 1, 1, 1, 1, 1, 1); });
    Tensor xd = rand_tensor({1, 3, 5, 6});
    check("dwconv_w", rand_tensor({3, 1, 3, 3}, 0.3f),
          [xd, bd](const Tensor& w) { return gr::dwconv2d(xd, w, bd, 1, 1, 1, 1, 1, 1); });
    check("dwconv_b", rand_tensor({3}, 0.3f),
          [xd, wd](const Tensor& b) { return gr::dwconv2d(xd, wd, b, 1, 1, 1, 1, 1, 1); });
    check("dwconv_s2", rand_tensor({1, 3, 6, 6}),
          [wd, bd](const Tensor& x) { return gr::dwconv2d(x, wd, bd, 2, 2, 1, 1, 1, 1); });
  }

  printf("%s\n", failures ? "grad_check: FAIL" : "grad_check: OK");
  return failures ? 1 : 0;
}
