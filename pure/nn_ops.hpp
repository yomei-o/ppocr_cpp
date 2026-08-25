// Tensor ops the engine did not already have, lifted out of onnx_run.hpp.
//
// Plain maths: no graph, no node names, no value stores. They live here because two files need
// them — the interpreter that runs a graph forward, and onnx_grad.hpp which wraps each one with a
// backward. With them still inside onnx_run.hpp that second dependency would be circular, since
// onnx_run.hpp is itself the caller of the gradient wrappers.
//
// The block below is unchanged from where it came from; see the comments in it for why each op
// exists (the short version: PP-OCRv5's recogniser needs per-axis strides and pads that the
// engine's square-everything conv cannot express).
#pragma once
#include "autograd.hpp"
#include "ew.hpp"
#include "nd.hpp"
#include "parallel.hpp"
#include "backend.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Strip a tensor back to a plain buffer: no parents, no closure, no grad storage.
//
// Moved here with the ops because they all end in `return nograd(o)` — the interpreter's forward
// path wants neither a tape nor a grad buffer, and on these graphs the allocate-and-zero of grad
// measured as more than the arithmetic. onnx_grad.hpp deliberately does NOT call it: that is the
// whole difference between the two paths.
inline const Tensor& nograd(const Tensor& t) {
  if (t) {
    t->parents.clear();
    t->backward_fn = nullptr;
    t->grad.clear();
    t->grad.shrink_to_fit();
    t->requires_grad = false;
  }
  return t;
}

// ---- ONNX ops the engine did not already have --------------------------------------------------

// Conv with independent stride/pad/kernel per axis. The engine's own conv2d only takes one stride
// and one pad, which is fine for square-everything detectors but not for a text recognizer: this
// one downsamples (2,1) and (1,2) to squeeze height while keeping the time axis, and pads a 1x3
// kernel as (0,1,0,1). im2col + one GEMM per group, so it still lands on bk::gemm (Eigen/SIMD).
inline Tensor conv2d_gen(const Tensor& x, const Tensor& w, const Tensor& bias,
                         int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t pb, int64_t pr,
                         int64_t groups) {
  int64_t N = x->shape[0], Cin = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t Cout = w->shape[0], kh = w->shape[2], kw = w->shape[3];
  int64_t OH = (H + pt + pb - kh) / sh + 1;
  int64_t OW = (W + pl + pr - kw) / sw + 1;
  if (OH <= 0 || OW <= 0) { printf("conv2d_gen: empty output\n"); std::exit(1); }
  int64_t Cin_g = Cin / groups, Cout_g = Cout / groups;
  int64_t K = Cin_g * kh * kw, P = OH * OW;
  Tensor o = make_tensor({N, Cout, OH, OW}, false);
  std::vector<float> col((size_t)K * P);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t g = 0; g < groups; ++g) {
      const float* xg = x->data.data() + (n * Cin + g * Cin_g) * H * W;
      for (int64_t c = 0; c < Cin_g; ++c)
        for (int64_t r = 0; r < kh; ++r)
          for (int64_t s = 0; s < kw; ++s) {
            float* crow = col.data() + (((c * kh + r) * kw + s) * P);
            for (int64_t oh = 0; oh < OH; ++oh) {
              int64_t ih = oh * sh - pt + r;
              float* dst = crow + oh * OW;
              if (ih < 0 || ih >= H) { for (int64_t ow = 0; ow < OW; ++ow) dst[ow] = 0.f; continue; }
              const float* src = xg + (c * H + ih) * W;
              for (int64_t ow = 0; ow < OW; ++ow) {
                int64_t iw = ow * sw - pl + s;
                dst[ow] = (iw < 0 || iw >= W) ? 0.f : src[iw];
              }
            }
          }
      bk::gemm_hosted(w->data.data() + (g * Cout_g) * K, col.data(),
                      o->data.data() + (n * Cout + g * Cout_g) * P, Cout_g, K, P, 0.f);
    }
  if (bias)
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < Cout; ++c) {
        float b = bias->data[c];
        float* p = o->data.data() + (n * Cout + c) * P;
        for (int64_t i = 0; i < P; ++i) p[i] += b;
      }
  return nograd(o);
}

// Depthwise (groups == Cin == Cout). im2col would make one GEMM of M=1 per channel; the direct
// loop is both faster and allocation-free, and these nets are mostly depthwise.
inline Tensor dwconv2d_gen(const Tensor& x, const Tensor& w, const Tensor& bias,
                           int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t pb, int64_t pr) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t kh = w->shape[2], kw = w->shape[3];
  int64_t OH = (H + pt + pb - kh) / sh + 1;
  int64_t OW = (W + pl + pr - kw) / sw + 1;
  Tensor o = make_tensor({N, C, OH, OW}, false);
  const float* B = bias ? bias->data.data() : nullptr;
  for (int64_t n = 0; n < N; ++n)
    parallel_for(C, [&](int64_t c) {
      const float* xp = x->data.data() + (n * C + c) * H * W;
      const float* wp = w->data.data() + c * kh * kw;
      float* op = o->data.data() + (n * C + c) * OH * OW;
      float b = B ? B[c] : 0.f;
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow) {
          float s = b;
          for (int64_t r = 0; r < kh; ++r) {
            int64_t ih = oh * sh - pt + r;
            if (ih < 0 || ih >= H) continue;
            const float* xr = xp + ih * W;
            const float* wr = wp + r * kw;
            for (int64_t t = 0; t < kw; ++t) {
              int64_t iw = ow * sw - pl + t;
              if (iw < 0 || iw >= W) continue;
              s += xr[iw] * wr[t];
            }
          }
          op[oh * OW + ow] = s;
        }
    });
  return nograd(o);
}

// ConvTranspose (a.k.a. deconv). The DB head upsamples twice with stride-2 2x2 kernels.
// ONNX weight layout is [Cin, Cout/groups, kh, kw] — note this is NOT Conv's [Cout, Cin/g, kh, kw].
inline Tensor conv_transpose2d(const Tensor& x, const Tensor& w, const Tensor& bias,
                               int64_t sh, int64_t sw, int64_t pt, int64_t pl,
                               int64_t pb, int64_t pr, int64_t opad_h, int64_t opad_w,
                               int64_t groups) {
  int64_t N = x->shape[0], Cin = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t Cout_g = w->shape[1], kh = w->shape[2], kw = w->shape[3];
  int64_t Cin_g = Cin / groups, Cout = Cout_g * groups;
  int64_t OH = (H - 1) * sh + opad_h + kh - pt - pb;
  int64_t OW = (W - 1) * sw + opad_w + kw - pl - pr;
  Tensor o = make_tensor({N, Cout, OH, OW}, false);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t g = 0; g < groups; ++g)
      for (int64_t ci = 0; ci < Cin_g; ++ci) {
        int64_t cin = g * Cin_g + ci;
        const float* xp = &x->data[(n * Cin + cin) * H * W];
        for (int64_t co = 0; co < Cout_g; ++co) {
          const float* wp = &w->data[(cin * Cout_g + co) * kh * kw];
          float* op = &o->data[(n * Cout + g * Cout_g + co) * OH * OW];
          for (int64_t h = 0; h < H; ++h)
            for (int64_t r = 0; r < kh; ++r) {
              int64_t oh = h * sh + r - pt;
              if (oh < 0 || oh >= OH) continue;
              const float* wr = wp + r * kw;
              float* orow = op + oh * OW;
              for (int64_t ww = 0; ww < W; ++ww) {
                float v = xp[h * W + ww];
                if (v == 0.f) continue;
                for (int64_t s = 0; s < kw; ++s) {
                  int64_t ow = ww * sw + s - pl;
                  if (ow < 0 || ow >= OW) continue;
                  orow[ow] += v * wr[s];
                }
              }
            }
        }
      }
  if (bias)
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < Cout; ++c) {
        float b = bias->data[c];
        float* p = &o->data[(n * Cout + c) * OH * OW];
        for (int64_t i = 0; i < OH * OW; ++i) p[i] += b;
      }
  return nograd(o);
}

// AveragePool with independent kernel/stride per axis (the recognizer's neck pools 3x2).
inline Tensor avgpool2d(const Tensor& x, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                        int64_t pt, int64_t pl, int64_t pb, int64_t pr,
                        bool count_include_pad, bool ceil_mode) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  auto outdim = [&](int64_t in, int64_t k, int64_t s, int64_t p0, int64_t p1) {
    double v = (double)(in + p0 + p1 - k) / (double)s;
    return (int64_t)(ceil_mode ? std::ceil(v) : std::floor(v)) + 1;
  };
  int64_t OH = outdim(H, kh, sh, pt, pb), OW = outdim(W, kw, sw, pl, pr);
  Tensor o = make_tensor({N, C, OH, OW}, false);
  for (int64_t nc = 0; nc < N * C; ++nc)
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        double s = 0; int64_t cnt = 0;
        for (int64_t r = 0; r < kh; ++r) {
          int64_t ih = oh * sh + r - pt;
          for (int64_t t = 0; t < kw; ++t) {
            int64_t iw = ow * sw + t - pl;
            bool inside = (ih >= 0 && ih < H && iw >= 0 && iw < W);
            if (inside) { s += x->data[(nc * H + ih) * W + iw]; ++cnt; }
            else if (count_include_pad) ++cnt;
          }
        }
        o->data[(nc * OH + oh) * OW + ow] = cnt ? (float)(s / cnt) : 0.f;
      }
  return nograd(o);
}

// Nearest resize, ONNX asymmetric + floor (what the FPN exports use).
inline Tensor resize_nearest(const Tensor& x, int64_t OH, int64_t OW) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  Tensor o = make_tensor({N, C, OH, OW}, false);
  std::vector<int64_t> mapw((size_t)OW);
  for (int64_t ow = 0; ow < OW; ++ow) {
    int64_t iw = (int64_t)std::floor((double)ow * (double)W / (double)OW);
    mapw[(size_t)ow] = iw < 0 ? 0 : (iw >= W ? W - 1 : iw);
  }
  for (int64_t nc = 0; nc < N * C; ++nc)
    for (int64_t oh = 0; oh < OH; ++oh) {
      int64_t ih = (int64_t)std::floor((double)oh * (double)H / (double)OH);
      ih = ih < 0 ? 0 : (ih >= H ? H - 1 : ih);
      const float* src = &x->data[(nc * H + ih) * W];
      float* dst = &o->data[(nc * OH + oh) * OW];
      for (int64_t ow = 0; ow < OW; ++ow) dst[ow] = src[mapw[(size_t)ow]];
    }
  return nograd(o);
}

// Mean over a set of axes, keepdims optional.
inline Tensor reduce_mean(const Tensor& x, std::vector<int64_t> axes, bool keepdims) {
  size_t r = x->shape.size();
  if (axes.empty()) { axes.resize(r); for (size_t i = 0; i < r; ++i) axes[i] = (int64_t)i; }
  std::vector<char> red(r, 0);
  for (int64_t a : axes) red[(size_t)nd::norm_axis(a, r)] = 1;
  std::vector<int64_t> out;
  for (size_t i = 0; i < r; ++i) {
    if (!red[i]) out.push_back(x->shape[i]);
    else if (keepdims) out.push_back(1);
  }
  if (out.empty()) out.push_back(1);
  nd::Shape sx = nd::strides_of(x->shape), so = nd::strides_of(out);
  int64_t n = x->numel(), cnt = 1;
  for (size_t i = 0; i < r; ++i) if (red[i]) cnt *= x->shape[i];
  Tensor o = make_tensor(out, false);
  std::vector<double> acc((size_t)o->numel(), 0.0);
  for (int64_t idx = 0; idx < n; ++idx) {
    int64_t rem = idx, dst = 0; size_t oi = 0;
    for (size_t k = 0; k < r; ++k) {
      int64_t c = rem / sx[k];
      rem -= c * sx[k];
      if (red[k]) { if (keepdims) ++oi; }
      else { dst += c * so[oi]; ++oi; }
    }
    acc[(size_t)dst] += x->data[idx];
  }
  const int64_t on = o->numel();
  for (int64_t i = 0; i < on; ++i) o->data[i] = (float)(acc[(size_t)i] / (double)cnt);
  return nograd(o);
}

// Batched matmul that routes each 2-D slice through bk::gemm (Eigen/SIMD when built with it).
// nd::matmul's plain triple loop is far slower here, and the CTC head alone is a
// [T,120] x [120,18385] product per text line — it dominates the pipeline if left naive.
inline Tensor matmul_nd(const Tensor& a, const Tensor& b) {
  size_t ra = a->shape.size(), rb = b->shape.size();
  if (ra < 2 || rb < 2) { printf("matmul_nd: rank < 2\n"); std::exit(1); }
  int64_t M = a->shape[ra - 2], K = a->shape[ra - 1];
  int64_t K2 = b->shape[rb - 2], N = b->shape[rb - 1];
  if (K != K2) { printf("matmul_nd: inner %lld vs %lld\n", (long long)K, (long long)K2); std::exit(1); }
  int64_t ba = 1, bb = 1;
  for (size_t i = 0; i + 2 < ra; ++i) ba *= a->shape[i];
  for (size_t i = 0; i + 2 < rb; ++i) bb *= b->shape[i];
  if (ba != bb && ba != 1 && bb != 1) {
    printf("matmul_nd: batch %lld vs %lld\n", (long long)ba, (long long)bb); std::exit(1);
  }
  int64_t batch = ba > bb ? ba : bb;
  std::vector<int64_t> out;
  const std::vector<int64_t>& lead = (ra >= rb ? a->shape : b->shape);
  for (size_t i = 0; i + 2 < lead.size(); ++i) out.push_back(lead[i]);
  out.push_back(M); out.push_back(N);
  Tensor o = make_tensor(out, false);
  for (int64_t t = 0; t < batch; ++t)
    bk::gemm_hosted(a->data.data() + (ba == 1 ? 0 : t) * M * K,
                    b->data.data() + (bb == 1 ? 0 : t) * K * N,
                    o->data.data() + t * M * N, M, K, N, 0.f);
  return nograd(o);
}
