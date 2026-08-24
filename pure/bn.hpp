// BatchNorm2d as a trainable op on the pure autograd engine (the fused-conv path is
// inference-only; this keeps conv/BN separate so weights can be trained and written
// back to a .pt). NCHW, per-channel affine. Matches torch.nn.BatchNorm2d numerics:
// normalization uses the biased batch variance; running_var is updated with the
// unbiased estimate. running_mean/var are buffers, updated in place in training mode.
#pragma once
#include "autograd.hpp"
#include <memory>

inline Tensor batchnorm2d(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                          std::vector<float>& running_mean, std::vector<float>& running_var,
                          float eps = 1e-5f, bool training = true, float momentum = 0.1f) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t HW = H * W, M = N * HW;
  auto o = make_tensor(x->shape, true);

  std::vector<float> mean(C), rstd(C);
  if (training) {
    for (int64_t c = 0; c < C; ++c) {
      double s = 0;
      for (int64_t n = 0; n < N; ++n) for (int64_t i = 0; i < HW; ++i) s += x->data[(n * C + c) * HW + i];
      double mu = s / M, sv = 0;
      for (int64_t n = 0; n < N; ++n) for (int64_t i = 0; i < HW; ++i) { double d = x->data[(n * C + c) * HW + i] - mu; sv += d * d; }
      double v = sv / M;                                   // biased -> used to normalize
      mean[c] = (float)mu; rstd[c] = (float)(1.0 / std::sqrt(v + eps));
      running_mean[c] = (1 - momentum) * running_mean[c] + momentum * (float)mu;
      double vunb = M > 1 ? sv / (M - 1) : v;              // unbiased -> running_var
      running_var[c] = (1 - momentum) * running_var[c] + momentum * (float)vunb;
    }
  } else {
    for (int64_t c = 0; c < C; ++c) { mean[c] = running_mean[c]; rstd[c] = (float)(1.0 / std::sqrt(running_var[c] + eps)); }
  }

  for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) {
    float g = gamma->data[c], b = beta->data[c], mu = mean[c], rs = rstd[c];
    for (int64_t i = 0; i < HW; ++i) { int64_t idx = (n * C + c) * HW + i; o->data[idx] = g * (x->data[idx] - mu) * rs + b; }
  }

  o->parents = {x, gamma, beta};
  Node* op = o.get();
  auto meanp = std::make_shared<std::vector<float>>(std::move(mean));
  auto rstdp = std::make_shared<std::vector<float>>(std::move(rstd));
  o->backward_fn = [x, gamma, beta, op, N, C, HW, M, meanp, rstdp, training] {
    auto& mean = *meanp; auto& rstd = *rstdp;
    for (int64_t c = 0; c < C; ++c) {
      float g = gamma->data[c], mu = mean[c], rs = rstd[c];
      double sum_dy = 0, sum_dy_xhat = 0;
      for (int64_t n = 0; n < N; ++n) for (int64_t i = 0; i < HW; ++i) {
        int64_t idx = (n * C + c) * HW + i;
        float dy = op->grad[idx], xhat = (x->data[idx] - mu) * rs;
        sum_dy += dy; sum_dy_xhat += (double)dy * xhat;
      }
      beta->grad[c] += (float)sum_dy; gamma->grad[c] += (float)sum_dy_xhat;
      if (training) {
        float inv = 1.f / (float)M;
        for (int64_t n = 0; n < N; ++n) for (int64_t i = 0; i < HW; ++i) {
          int64_t idx = (n * C + c) * HW + i;
          float dy = op->grad[idx], xhat = (x->data[idx] - mu) * rs;
          x->grad[idx] += g * rs * (dy - inv * (float)sum_dy - xhat * inv * (float)sum_dy_xhat);
        }
      } else {
        for (int64_t n = 0; n < N; ++n) for (int64_t i = 0; i < HW; ++i) {
          int64_t idx = (n * C + c) * HW + i;
          x->grad[idx] += g * rs * op->grad[idx];
        }
      }
    }
  };
  return o;
}
