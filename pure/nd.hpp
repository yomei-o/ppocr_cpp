// Forward-only N-dimensional ops for the ONNX interpreter.
//
// The engine inherited from the sibling repos is built around 4D NCHW activations and 2D matrices,
// which is all a plain conv net needs. Modern detector graphs are not that shape: a YOLO12 export
// reshapes to [1,C,N], transposes for attention, batches MatMul over heads, and softmaxes over the
// last axis. Rather than bend the 4D ops, this header adds generic rank-agnostic versions.
//
// **Inference only.** These build no backward closure — training paths must keep using the 4D/2D
// ops in autograd.hpp / ops2d.hpp / linalg.hpp. onnx_run.hpp prefers the original ops whenever the
// shapes fit them, so the numerically verified paths (recognizer parity 3.3e-05, YOLOX decode) stay
// exactly as they were.
#pragma once
#include "autograd.hpp"
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace nd {

using Shape = std::vector<int64_t>;

inline int64_t numel(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

inline Shape strides_of(const Shape& s) {
  Shape st(s.size(), 1);
  for (int i = (int)s.size() - 2; i >= 0; --i) st[i] = st[i + 1] * s[i + 1];
  return st;
}

inline int64_t norm_axis(int64_t axis, size_t rank) {
  return axis < 0 ? axis + (int64_t)rank : axis;
}

// ---- elementwise binary with NumPy-style broadcasting ---------------------------------------
inline Tensor binary(const Tensor& a, const Tensor& b, const std::function<float(float, float)>& f) {
  const Shape& sa = a->shape;
  const Shape& sb = b->shape;
  size_t r = std::max(sa.size(), sb.size());
  Shape pa(r, 1), pb(r, 1), out(r, 1);
  for (size_t i = 0; i < sa.size(); ++i) pa[r - sa.size() + i] = sa[i];
  for (size_t i = 0; i < sb.size(); ++i) pb[r - sb.size() + i] = sb[i];
  for (size_t i = 0; i < r; ++i) {
    if (pa[i] != pb[i] && pa[i] != 1 && pb[i] != 1) {
      printf("nd::binary: shapes not broadcastable (axis %zu: %lld vs %lld)\n", i,
             (long long)pa[i], (long long)pb[i]);
      std::exit(1);
    }
    out[i] = std::max(pa[i], pb[i]);
  }
  Shape so = strides_of(out), sta = strides_of(pa), stb = strides_of(pb);
  for (size_t i = 0; i < r; ++i) {                        // a stride of 0 = broadcast that axis
    if (pa[i] == 1) sta[i] = 0;
    if (pb[i] == 1) stb[i] = 0;
  }
  Tensor o = make_tensor(out, false);
  int64_t n = numel(out);
  const float* da = a->data.data();
  const float* db = b->data.data();
  float* po = o->data.data();
  for (int64_t idx = 0; idx < n; ++idx) {
    int64_t rem = idx, ia = 0, ib = 0;
    for (size_t k = 0; k < r; ++k) {
      int64_t c = rem / so[k];
      rem -= c * so[k];
      ia += c * sta[k];
      ib += c * stb[k];
    }
    po[idx] = f(da[ia], db[ib]);
  }
  return o;
}

inline Tensor add(const Tensor& a, const Tensor& b) { return binary(a, b, [](float x, float y) { return x + y; }); }
inline Tensor sub(const Tensor& a, const Tensor& b) { return binary(a, b, [](float x, float y) { return x - y; }); }
inline Tensor mul(const Tensor& a, const Tensor& b) { return binary(a, b, [](float x, float y) { return x * y; }); }
inline Tensor div(const Tensor& a, const Tensor& b) { return binary(a, b, [](float x, float y) { return x / y; }); }

// ---- transpose with an arbitrary permutation --------------------------------------------------
inline Tensor transpose(const Tensor& x, const Shape& perm_in) {
  size_t r = x->shape.size();
  Shape perm = perm_in;
  if (perm.empty()) {                                     // ONNX default: reverse the axes
    perm.resize(r);
    for (size_t i = 0; i < r; ++i) perm[i] = (int64_t)(r - 1 - i);
  }
  Shape out(r);
  for (size_t i = 0; i < r; ++i) out[i] = x->shape[perm[i]];
  Shape sx = strides_of(x->shape), so = strides_of(out);
  Tensor o = make_tensor(out, false);
  int64_t n = numel(out);
  const float* d = x->data.data();
  float* p = o->data.data();
  for (int64_t idx = 0; idx < n; ++idx) {
    int64_t rem = idx, src = 0;
    for (size_t k = 0; k < r; ++k) {
      int64_t c = rem / so[k];
      rem -= c * so[k];
      src += c * sx[perm[k]];
    }
    p[idx] = d[src];
  }
  return o;
}

// ---- softmax over one axis --------------------------------------------------------------------
inline Tensor softmax(const Tensor& x, int64_t axis_in) {
  size_t r = x->shape.size();
  int64_t axis = norm_axis(axis_in, r);
  Shape st = strides_of(x->shape);
  int64_t n = numel(x->shape), len = x->shape[axis], step = st[axis];
  Tensor o = make_tensor(x->shape, false);
  const float* d = x->data.data();
  float* p = o->data.data();
  int64_t outer = n / len;
  for (int64_t k = 0; k < outer; ++k) {
    // map k to the base offset of a lane along `axis`
    int64_t rem = k, base = 0;
    for (size_t a = 0; a < r; ++a) {
      if ((int64_t)a == axis) continue;
      int64_t dim = x->shape[a];
      // iterate the non-axis dims in row-major order
      int64_t inner = 1;
      for (size_t b = a + 1; b < r; ++b) if ((int64_t)b != axis) inner *= x->shape[b];
      int64_t c = (rem / inner) % dim;
      base += c * st[a];
    }
    float m = -1e30f;
    for (int64_t i = 0; i < len; ++i) m = std::max(m, d[base + i * step]);
    float s = 0;
    for (int64_t i = 0; i < len; ++i) { float e = std::exp(d[base + i * step] - m); p[base + i * step] = e; s += e; }
    for (int64_t i = 0; i < len; ++i) p[base + i * step] /= s;
  }
  return o;
}

// ---- batched matmul over the last two dims ----------------------------------------------------
inline Tensor matmul(const Tensor& a, const Tensor& b) {
  size_t ra = a->shape.size(), rb = b->shape.size();
  if (ra < 2 || rb < 2) { printf("nd::matmul: rank < 2\n"); std::exit(1); }
  int64_t M = a->shape[ra - 2], K = a->shape[ra - 1];
  int64_t K2 = b->shape[rb - 2], N = b->shape[rb - 1];
  if (K != K2) { printf("nd::matmul: inner dims %lld vs %lld\n", (long long)K, (long long)K2); std::exit(1); }
  int64_t batch_a = 1, batch_b = 1;
  for (size_t i = 0; i + 2 < ra; ++i) batch_a *= a->shape[i];
  for (size_t i = 0; i + 2 < rb; ++i) batch_b *= b->shape[i];
  int64_t batch = std::max(batch_a, batch_b);
  Shape out;
  const Shape& lead = (ra >= rb ? a->shape : b->shape);
  for (size_t i = 0; i + 2 < lead.size(); ++i) out.push_back(lead[i]);
  out.push_back(M);
  out.push_back(N);
  Tensor o = make_tensor(out, false);
  const float* da = a->data.data();
  const float* db = b->data.data();
  float* po = o->data.data();
  for (int64_t bt = 0; bt < batch; ++bt) {
    const float* A = da + (batch_a == 1 ? 0 : bt) * M * K;
    const float* B = db + (batch_b == 1 ? 0 : bt) * K * N;
    float* O = po + bt * M * N;
    for (int64_t i = 0; i < M; ++i)
      for (int64_t k = 0; k < K; ++k) {
        float av = A[i * K + k];
        if (av == 0.f) continue;
        const float* Br = B + k * N;
        float* Or = O + i * N;
        for (int64_t j = 0; j < N; ++j) Or[j] += av * Br[j];
      }
  }
  return o;
}

// ---- slice / concat / split over an arbitrary axis --------------------------------------------
inline Tensor slice(const Tensor& x, const Shape& starts, const Shape& ends,
                    const Shape& axes_in, const Shape& steps_in) {
  size_t r = x->shape.size();
  Shape axes = axes_in;
  if (axes.empty()) { axes.resize(starts.size()); for (size_t i = 0; i < axes.size(); ++i) axes[i] = (int64_t)i; }
  Shape steps = steps_in;
  if (steps.empty()) steps.assign(axes.size(), 1);
  Shape beg(r, 0), stp(r, 1), out = x->shape;
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t a = norm_axis(axes[i], r), dim = x->shape[a];
    int64_t s = starts[i] < 0 ? starts[i] + dim : starts[i];
    int64_t e = ends[i] < 0 ? ends[i] + dim : ends[i];
    s = std::max<int64_t>(0, std::min(s, dim));
    e = std::max<int64_t>(0, std::min(e, dim));
    beg[a] = s;
    stp[a] = steps[i];
    out[a] = (e - s + steps[i] - 1) / steps[i];
  }
  Shape sx = strides_of(x->shape), so = strides_of(out);
  Tensor o = make_tensor(out, false);
  int64_t n = numel(out);
  const float* d = x->data.data();
  float* p = o->data.data();
  for (int64_t idx = 0; idx < n; ++idx) {
    int64_t rem = idx, src = 0;
    for (size_t k = 0; k < r; ++k) {
      int64_t c = rem / so[k];
      rem -= c * so[k];
      src += (beg[k] + c * stp[k]) * sx[k];
    }
    p[idx] = d[src];
  }
  return o;
}

inline Tensor concat(const std::vector<Tensor>& xs, int64_t axis_in) {
  size_t r = xs[0]->shape.size();
  int64_t axis = norm_axis(axis_in, r);
  Shape out = xs[0]->shape;
  out[axis] = 0;
  for (const Tensor& t : xs) out[axis] += t->shape[axis];
  Tensor o = make_tensor(out, false);
  Shape so = strides_of(out);
  int64_t off = 0;
  for (const Tensor& t : xs) {
    Shape st = strides_of(t->shape);
    int64_t n = numel(t->shape);
    const float* d = t->data.data();
    float* p = o->data.data();
    for (int64_t idx = 0; idx < n; ++idx) {
      int64_t rem = idx, dst = 0;
      for (size_t k = 0; k < r; ++k) {
        int64_t c = rem / st[k];
        rem -= c * st[k];
        dst += ((int64_t)k == axis ? c + off : c) * so[k];
      }
      p[dst] = d[idx];
    }
    off += t->shape[axis];
  }
  return o;
}

inline std::vector<Tensor> split(const Tensor& x, int64_t axis_in, const Shape& sizes) {
  size_t r = x->shape.size();
  int64_t axis = norm_axis(axis_in, r);
  std::vector<Tensor> outs;
  int64_t off = 0;
  for (int64_t s : sizes) {
    Shape st{off}, en{off + s}, ax{axis}, sp{1};
    outs.push_back(slice(x, st, en, ax, sp));
    off += s;
  }
  return outs;
}

}  // namespace nd
