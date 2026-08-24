// FaceNet-specific differentiable ops not in the shared engine: ReLU, asymmetric spatial pad
// (Inception 1x7 / 7x1 convs), global average pool, dense linear (matmul+bias), and row L2
// normalize (F.normalize p=2 dim=1). Same backward-registration convention as autograd.hpp.
#pragma once
#include "autograd.hpp"
#include "linalg.hpp"      // matmul, transpose2d
#include <cmath>
#include <algorithm>

inline Tensor relu(const Tensor& a) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i = 0; i < a->numel(); ++i) o->data[i] = a->data[i] > 0 ? a->data[i] : 0.f;
  o->parents = {a}; Node* op = o.get();
  o->backward_fn = [a, op] { for (int64_t i = 0; i < op->numel(); ++i) if (a->data[i] > 0) a->grad[i] += op->grad[i]; };
  return o;
}

// zero-pad H by ph each side, W by pw each side (symmetric per dim, PyTorch-style).
inline Tensor pad_hw(const Tensor& x, int64_t ph, int64_t pw) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t OH = H + 2 * ph, OW = W + 2 * pw;
  auto o = make_tensor({N, C, OH, OW}, true);
  for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c)
    for (int64_t h = 0; h < H; ++h) {
      const float* src = &x->data[((n * C + c) * H + h) * W];
      float* dst = &o->data[((n * C + c) * OH + (h + ph)) * OW + pw];
      for (int64_t w = 0; w < W; ++w) dst[w] = src[w];
    }
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, N, C, H, W, OH, OW, ph, pw] {
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c)
      for (int64_t h = 0; h < H; ++h) {
        float* gx = &x->grad[((n * C + c) * H + h) * W];
        const float* go = &op->grad[((n * C + c) * OH + (h + ph)) * OW + pw];
        for (int64_t w = 0; w < W; ++w) gx[w] += go[w];
      }
  };
  return o;
}

// conv with per-dim (symmetric) padding: pad then call the shared square-pad conv2d with pad=0.
inline Tensor conv2d_hw(const Tensor& x, const Tensor& w, const Tensor& b,
                        int64_t stride, int64_t ph, int64_t pw) {
  Tensor px = (ph || pw) ? pad_hw(x, ph, pw) : x;
  return conv2d(px, w, b, stride, 0, 1);
}

// global average pool: (N,C,H,W) -> (N,C).
inline Tensor gap(const Tensor& x) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], HW = H * W;
  auto o = make_tensor({N, C}, true);
  for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) {
    double s = 0; const float* p = &x->data[(n * C + c) * HW];
    for (int64_t i = 0; i < HW; ++i) s += p[i];
    o->data[n * C + c] = (float)(s / HW);
  }
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, N, C, HW] {
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) {
      float g = op->grad[n * C + c] / (float)HW; float* gx = &x->grad[(n * C + c) * HW];
      for (int64_t i = 0; i < HW; ++i) gx[i] += g;
    }
  };
  return o;
}

// add a per-column bias vector to every row: y[n,d] = x[n,d] + b[d].
inline Tensor add_rowvec(const Tensor& x, const Tensor& b) {
  int64_t N = x->shape[0], D = x->shape[1];
  auto o = make_tensor(x->shape, true);
  for (int64_t n = 0; n < N; ++n) for (int64_t d = 0; d < D; ++d) o->data[n * D + d] = x->data[n * D + d] + b->data[d];
  o->parents = {x, b}; Node* op = o.get();
  o->backward_fn = [x, b, op, N, D] {
    for (int64_t n = 0; n < N; ++n) for (int64_t d = 0; d < D; ++d) {
      x->grad[n * D + d] += op->grad[n * D + d]; b->grad[d] += op->grad[n * D + d];
    }
  };
  return o;
}

// dense layer: x[N,in], W[out,in], b[out] -> [N,out]  (y = x·Wᵀ + b).
inline Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b) {
  return add_rowvec(matmul(x, transpose2d(W)), b);
}

// row L2 normalize: y = x / max(‖x‖₂, eps)  (matches F.normalize p=2 dim=1).
inline Tensor l2norm_rows(const Tensor& x, float eps = 1e-12f) {
  int64_t N = x->shape[0], D = x->shape[1];
  auto o = make_tensor(x->shape, true);
  std::vector<float> inv(N);
  for (int64_t n = 0; n < N; ++n) {
    double ss = 0; const float* p = &x->data[n * D]; for (int64_t d = 0; d < D; ++d) ss += (double)p[d] * p[d];
    float nrm = (float)std::sqrt(ss); inv[n] = 1.f / std::max(nrm, eps);
    float* q = &o->data[n * D]; for (int64_t d = 0; d < D; ++d) q[d] = p[d] * inv[n];
  }
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, N, D, inv] {
    for (int64_t n = 0; n < N; ++n) {
      const float* y = &op->data[n * D]; const float* gy = &op->grad[n * D]; float* gx = &x->grad[n * D];
      double dot = 0; for (int64_t d = 0; d < D; ++d) dot += (double)y[d] * gy[d];
      for (int64_t d = 0; d < D; ++d) gx[d] += inv[n] * (gy[d] - y[d] * (float)dot);
    }
  };
  return o;
}
