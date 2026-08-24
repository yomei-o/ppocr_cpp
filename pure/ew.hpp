// Forward-only elementwise ops, written to actually vectorize.
//
// The engine inherited from the sibling repos writes its elementwise ops as
//
//     for (int64_t i = 0; i < o->numel(); ++i) o->data[i] = a->data[i] + b->data[i];
//
// and Node::numel() is a std::accumulate over the shape vector. Re-evaluating it every iteration is
// a multiply-accumulate per element and, worse, leaves the compiler unable to treat the bound as
// loop-invariant, so the loop stays scalar. Measured on this repo's detector that came to ~20 ns
// per element: the 117 Add nodes cost 480 ms, more than every convolution but one. Those ops are
// correct and they carry the autograd tape the training code in the sibling repos needs, so they
// are left alone; the interpreter routes through these instead, which hoist the bound and work on
// raw pointers.
//
// exp() comes from fastmath.hpp for the reason documented there.
#pragma once
#include "autograd.hpp"
#include "fastmath.hpp"
#include "parallel.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

namespace ew {

inline int64_t count(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

inline Tensor like(const std::vector<int64_t>& shape) { return make_tensor(shape, false); }

template <class F>
inline Tensor unary(const Tensor& x, F f) {
  Tensor o = like(x->shape);
  const int64_t n = count(x->shape);
  const float* __restrict a = x->data.data();
  float* __restrict p = o->data.data();
  for (int64_t i = 0; i < n; ++i) p[i] = f(a[i]);
  return o;
}

template <class F>
inline Tensor binary_same(const Tensor& x, const Tensor& y, F f) {
  Tensor o = like(x->shape);
  const int64_t n = count(x->shape);
  const float* __restrict a = x->data.data();
  const float* __restrict b = y->data.data();
  float* __restrict p = o->data.data();
  for (int64_t i = 0; i < n; ++i) p[i] = f(a[i], b[i]);
  return o;
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  return binary_same(a, b, [](float x, float y) { return x + y; });
}
inline Tensor mul(const Tensor& a, const Tensor& b) {
  return binary_same(a, b, [](float x, float y) { return x * y; });
}
inline Tensor add_scalar(const Tensor& a, float c) {
  return unary(a, [c](float x) { return x + c; });
}
inline Tensor mul_scalar(const Tensor& a, float c) {
  return unary(a, [c](float x) { return x * c; });
}
inline Tensor relu(const Tensor& a) {
  return unary(a, [](float x) { return x > 0.f ? x : 0.f; });
}
inline Tensor sigmoid(const Tensor& a) {
  return unary(a, [](float x) { return fm::sigmoid_(x); });
}
inline Tensor hardswish(const Tensor& a) {
  return unary(a, [](float x) {
    float t = x / 6.f + 0.5f;
    return x * (t < 0.f ? 0.f : (t > 1.f ? 1.f : t));
  });
}
inline Tensor hardsigmoid(const Tensor& a, float alpha, float beta) {
  return unary(a, [alpha, beta](float x) {
    float t = alpha * x + beta;
    return t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  });
}

// Reshape without touching the data (the interpreter reshapes constantly and the vendored reshape
// copies through the same numel-in-the-condition loop).
inline Tensor reshape(const Tensor& x, std::vector<int64_t> shape) {
  Tensor o = std::make_shared<Node>();
  o->shape = std::move(shape);
  o->data = x->data;
  return o;
}

// Inference batch norm, folded to one multiply-add per channel.
inline Tensor batchnorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                        const Tensor& mean, const Tensor& var, float eps) {
  const int64_t N = x->shape[0], C = x->shape[1];
  const int64_t HW = count(x->shape) / (N * C);
  Tensor o = like(x->shape);
  std::vector<float> sc((size_t)C), sh((size_t)C);
  for (int64_t c = 0; c < C; ++c) {
    float s = gamma->data[(size_t)c] / std::sqrt(var->data[(size_t)c] + eps);
    sc[(size_t)c] = s;
    sh[(size_t)c] = beta->data[(size_t)c] - mean->data[(size_t)c] * s;
  }
  for (int64_t nc = 0; nc < N * C; ++nc) {
    const float s = sc[(size_t)(nc % C)], b = sh[(size_t)(nc % C)];
    const float* __restrict src = x->data.data() + nc * HW;
    float* __restrict dst = o->data.data() + nc * HW;
    for (int64_t i = 0; i < HW; ++i) dst[i] = src[i] * s + b;
  }
  return o;
}

// Global average pool -> [N,C,1,1].
inline Tensor gap(const Tensor& x) {
  const int64_t N = x->shape[0], C = x->shape[1];
  const int64_t HW = count(x->shape) / (N * C);
  Tensor o = like({N, C, 1, 1});
  for (int64_t nc = 0; nc < N * C; ++nc) {
    const float* p = x->data.data() + nc * HW;
    double s = 0;
    for (int64_t i = 0; i < HW; ++i) s += p[i];
    o->data[(size_t)nc] = (float)(s / (double)HW);
  }
  return o;
}

// Softmax over the last axis — the only axis these graphs use. The rank-agnostic version pays an
// O(rank) index computation per element, which on a [1,T,18385] CTC head is the whole cost.
inline Tensor softmax_last(const Tensor& x) {
  const int64_t C = x->shape.back();
  const int64_t rows = count(x->shape) / C;
  Tensor o = like(x->shape);
  const float* d = x->data.data();
  float* p = o->data.data();
  for (int64_t r = 0; r < rows; ++r) {
    const float* __restrict src = d + r * C;
    float* __restrict dst = p + r * C;
    float m = src[0];
    for (int64_t i = 1; i < C; ++i) m = src[i] > m ? src[i] : m;
    float s = 0;
    for (int64_t i = 0; i < C; ++i) { float e = fm::exp_(src[i] - m); dst[i] = e; s += e; }
    const float inv = 1.f / s;
    for (int64_t i = 0; i < C; ++i) dst[i] *= inv;
  }
  return o;
}

// b broadcast over the channel axis of an NCHW a: [N,C,H,W] op [1,C,1,1] — a folded bias, an SE
// block's gate. Everywhere in both graphs.
template <class F>
inline Tensor bcast_channel(const Tensor& a, const Tensor& b, F f) {
  const int64_t N = a->shape[0], C = a->shape[1];
  const int64_t HW = count(a->shape) / (N * C);
  Tensor o = like(a->shape);
  const float* db = b->data.data();
  for (int64_t nc = 0; nc < N * C; ++nc) {
    const float bv = db[nc % C];
    const float* __restrict src = a->data.data() + nc * HW;
    float* __restrict dst = o->data.data() + nc * HW;
    for (int64_t i = 0; i < HW; ++i) dst[i] = f(src[i], bv);
  }
  return o;
}

// b broadcast over the trailing axis of a: [..., C] op [C] — a Linear's bias, a LayerNorm's gain.
template <class F>
inline Tensor bcast_trailing(const Tensor& a, const Tensor& b, F f) {
  const int64_t C = b->data.size() ? (int64_t)b->data.size() : 1;
  const int64_t rows = count(a->shape) / C;
  Tensor o = like(a->shape);
  const float* db = b->data.data();
  for (int64_t r = 0; r < rows; ++r) {
    const float* __restrict src = a->data.data() + r * C;
    float* __restrict dst = o->data.data() + r * C;
    for (int64_t i = 0; i < C; ++i) dst[i] = f(src[i], db[i]);
  }
  return o;
}

}  // namespace ew
