// Graph-driven ONNX interpreter for the PP-OCR models — no onnxruntime, no protobuf library.
// Reads the Graph that onnx.hpp parsed and executes it on the pure engine's ops.
//
// Three things here that the yolo_lpr_cpp ancestor of this file did not need:
//
//  1. AN INT64 VALUE PATH. PP-OCRv5's recognizer computes its own Reshape targets at runtime
//     (Shape -> Slice -> Concat -> Reshape) because the text-line width is dynamic. A float-only
//     interpreter cannot represent Shape's output at all, so shape math is carried in a second
//     store (iv) of int64 tensors, and Slice/Concat/Squeeze/Unsqueeze/Gather/Cast dispatch on
//     which store the operand lives in.
//
//  2. LIVENESS-BASED FREEING. Every weight in these exports is a Constant *node*, not an
//     initializer (paddle2onnx does that), so the graph is 16 MB of tensors and the activations at
//     960x960 are far bigger. Keeping every named value until the end costs sum-of-all-activations;
//     the interpreter instead counts consumers per value and erases one as soon as its last reader
//     has run. Peak memory becomes max-over-time, which is what makes the detector fit in WASM.
//
//  3. NO TAPE. autograd.hpp's ops all build a backward closure and hold their parents alive, which
//     would defeat (2), and allocate a grad buffer the same size as the data. nograd() strips both
//     right after each op — this engine is inference only.
#pragma once
#include "onnx.hpp"
#include "autograd.hpp"
#include "ops2d.hpp"       // reshape, mul_scalar, add_scalar, softmax_rows
#include "ops_yolox.hpp"   // dwconv2d
#include "face_ops.hpp"    // relu, conv2d_hw, pad_hw, gap, add_rowvec
#include "linalg.hpp"      // matmul, transpose2d
#include "bn.hpp"          // batchnorm2d
#include "nd.hpp"          // rank-agnostic transpose/softmax/slice/concat
#include "ew.hpp"          // forward-only elementwise (see that header: the tape ops do not vectorize)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace onx {

// ---- an int64 value (a shape, an axes list, a Slice bound) ----
struct IVal { std::vector<int64_t> dims; std::vector<int64_t> data; };

// Drop the tape: no grad buffer, no parents, no backward closure. Without this every intermediate
// stays reachable from its consumers and nothing can be freed early.
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

inline const Attr* find_attr(const Node& n, const std::string& name) {
  for (auto& a : n.attr) if (a.name == name) return &a;
  return nullptr;
}
inline int64_t attr_i(const Node& n, const std::string& name, int64_t def) {
  const Attr* a = find_attr(n, name);
  if (!a) return def;
  return a->ints.empty() ? a->i : a->ints[0];
}
inline float attr_f(const Node& n, const std::string& name, float def) {
  const Attr* a = find_attr(n, name);
  return a ? a->f : def;
}
inline std::vector<int64_t> attr_ints(const Node& n, const std::string& name) {
  const Attr* a = find_attr(n, name);
  return a ? a->ints : std::vector<int64_t>{};
}
inline std::string attr_s(const Node& n, const std::string& name, const std::string& def) {
  const Attr* a = find_attr(n, name);
  return a ? a->s : def;
}

// Does b broadcast purely over a channel axis / the trailing axis? (All other axes must be 1.)
// A 1-D b is NOT a channel broadcast: ONNX aligns shapes from the right, so [C] broadcasts over W.
// These graphs always Reshape a folded bias to [1,C,1,1] first, which is what this matches.
inline bool is_channel_bcast(const Tensor& a, const Tensor& b) {
  if (a->shape.size() != 4 || b->shape.size() != 4 || b->numel() <= 1) return false;
  return b->shape[0] == 1 && b->shape[1] == a->shape[1] && b->shape[2] == 1 && b->shape[3] == 1;
}
inline bool is_trailing_bcast(const Tensor& a, const Tensor& b) {
  if (b->numel() <= 1 || a->shape.empty() || b->numel() != a->shape.back()) return false;
  for (size_t i = 0; i + 1 < b->shape.size(); ++i) if (b->shape[i] != 1) return false;
  return b->shape.back() == b->numel();
}

// ---- ONNX ops the engine did not already have --------------------------------------------------

// Pad with independent begin/end on H and W (ONNX Conv pads = [t, l, b, r]).
inline Tensor pad_hw4(const Tensor& x, int64_t t, int64_t l, int64_t b, int64_t r) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t OH = H + t + b, OW = W + l + r;
  Tensor o = make_tensor({N, C, OH, OW}, false);
  for (int64_t nc = 0; nc < N * C; ++nc)
    for (int64_t h = 0; h < H; ++h) {
      const float* src = &x->data[(nc * H + h) * W];
      float* dst = &o->data[(nc * OH + (h + t)) * OW + l];
      for (int64_t w = 0; w < W; ++w) dst[w] = src[w];
    }
  return nograd(o);
}

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

// =============================== the interpreter ===============================

struct Run {
  std::map<std::string, Tensor> fv;   // float values
  std::map<std::string, IVal> iv;     // int64 values (shapes, axes, bounds)

  Tensor f(const std::string& n) const {
    auto it = fv.find(n);
    if (it == fv.end()) {
      fprintf(stderr, "onnx_run: float value '%s' missing (an earlier op unimplemented?)\n", n.c_str());
      std::exit(2);
    }
    return it->second;
  }
  const IVal& i(const std::string& n) const {
    auto it = iv.find(n);
    if (it == iv.end()) {
      fprintf(stderr, "onnx_run: int value '%s' missing\n", n.c_str());
      std::exit(2);
    }
    return it->second;
  }
  bool has_i(const std::string& n) const { return iv.count(n) != 0; }
  bool has_f(const std::string& n) const { return fv.count(n) != 0; }
};

// Materialized weights, shared across calls. paddle2onnx emits every weight as a Constant *node*,
// so without this the recognizer re-allocates, re-zeroes and re-copies 16 MB of tensors for every
// text line in the image — which for a page of 50 lines is most of the wall clock.
// Per-op-type timing, accumulated without printing (a printf per node costs more than the ops).
struct Prof {
  std::map<std::string, double> ms;
  std::map<std::string, int> n;
  void add(const std::string& op, double dt) { ms[op] += dt; ++n[op]; }
};

struct Weights {
  std::map<std::string, Tensor> f;
  std::map<std::string, IVal> i;
  bool empty() const { return f.empty() && i.empty(); }
};

inline Weights build_weights(const Graph& g) {
  Weights w;
  for (const auto& t : g.init_f) w.f[t.name] = nograd(from_data(t.dims, t.data));
  for (const auto& t : g.init_i) w.i[t.name] = IVal{t.dims, t.data};
  for (const auto& n : g.nodes) {
    if (n.op_type != "Constant") continue;
    const Attr* a = find_attr(n, "value");
    if (!a || !a->has_tensor) continue;
    if (a->t_dtype == 7) w.i[n.output[0]] = IVal{a->t_dims, a->t_ints};
    else {
      std::vector<int64_t> dims = a->t_dims;
      if (dims.empty()) dims.push_back((int64_t)a->t_floats.size());
      w.f[n.output[0]] = nograd(from_data(dims, a->t_floats));
    }
  }
  return w;
}

// Run g on the named inputs. want names the values to return (empty = the graph's declared
// outputs). Everything not named and no longer needed is freed as soon as its last reader ran.
inline std::map<std::string, Tensor> run_graph(const Graph& g,
                                              const std::map<std::string, Tensor>& inputs,
                                              const std::vector<std::string>& want = {},
                                              bool verbose = false,
                                              const Weights* wcache = nullptr,
                                              Prof* prof = nullptr) {
  infer_only() = true;                  // no op on this path ever needs a grad buffer
  Run R;
  for (auto& kv : inputs) R.fv[kv.first] = kv.second;
  if (wcache) {
    for (const auto& t : g.init_f) R.fv[t.name] = wcache->f.at(t.name);
    for (const auto& t : g.init_i) R.iv[t.name] = wcache->i.at(t.name);
  } else {
    for (const auto& t : g.init_f) R.fv[t.name] = nograd(from_data(t.dims, t.data));
    for (const auto& t : g.init_i) R.iv[t.name] = IVal{t.dims, t.data};
  }

  std::set<std::string> keep(want.begin(), want.end());
  if (keep.empty()) for (const auto& vi : g.outputs) keep.insert(vi.name);

  std::map<std::string, int> uses;
  for (const auto& n : g.nodes) for (const auto& s : n.input) if (!s.empty()) ++uses[s];

  auto put_f = [&](const std::string& name, const Tensor& t) { R.fv[name] = nograd(t); };
  double tprev = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;

  for (const auto& node : g.nodes) {
    double tnode = prof ? (double)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0
                        : 0.0;
    const std::string& op = node.op_type;
    const auto& in = node.input;
    const auto& outn = node.output;

    if (op == "Constant") {
      bool cached = false;
      if (wcache) {                                     // already materialized once, just alias it
        auto fi = wcache->f.find(outn[0]);
        if (fi != wcache->f.end()) { R.fv[outn[0]] = fi->second; cached = true; }
        else {
          auto ii = wcache->i.find(outn[0]);
          if (ii != wcache->i.end()) { R.iv[outn[0]] = ii->second; cached = true; }
        }
      }
      const Attr* a = cached ? nullptr : find_attr(node, "value");
      if (a && a->has_tensor) {
        if (a->t_dtype == 7) R.iv[outn[0]] = IVal{a->t_dims, a->t_ints};
        else {
          std::vector<int64_t> dims = a->t_dims;
          if (dims.empty()) dims.push_back((int64_t)a->t_floats.size());
          put_f(outn[0], from_data(dims, a->t_floats));
        }
      }
    } else if (op == "Conv") {
      Tensor x = R.f(in[0]), w = R.f(in[1]);
      Tensor b = (in.size() >= 3 && !in[2].empty()) ? R.f(in[2]) : nullptr;
      auto st = attr_ints(node, "strides"), pd = attr_ints(node, "pads"), dl = attr_ints(node, "dilations");
      int64_t sh = st.size() > 0 ? st[0] : 1, sw = st.size() > 1 ? st[1] : sh;
      int64_t pt = pd.size() > 0 ? pd[0] : 0, pl = pd.size() > 1 ? pd[1] : pt;
      int64_t pb = pd.size() > 2 ? pd[2] : pt, pr = pd.size() > 3 ? pd[3] : pl;
      int64_t grp = attr_i(node, "group", 1);
      for (int64_t d : dl) if (d != 1) { printf("Conv: dilation %lld unsupported\n", (long long)d); std::exit(1); }
      bool depthwise = grp > 1 && grp == w->shape[0] && w->shape[1] == 1;
      put_f(outn[0], depthwise ? dwconv2d_gen(x, w, b, sh, sw, pt, pl, pb, pr)
                               : conv2d_gen(x, w, b, sh, sw, pt, pl, pb, pr, grp));
    } else if (op == "ConvTranspose") {
      Tensor x = R.f(in[0]), w = R.f(in[1]);
      Tensor b = (in.size() >= 3 && !in[2].empty()) ? R.f(in[2]) : nullptr;
      auto st = attr_ints(node, "strides"), pd = attr_ints(node, "pads"), opd = attr_ints(node, "output_padding");
      int64_t sh = st.size() > 0 ? st[0] : 1, sw = st.size() > 1 ? st[1] : sh;
      int64_t pt = pd.size() > 0 ? pd[0] : 0, pl = pd.size() > 1 ? pd[1] : pt;
      int64_t pb = pd.size() > 2 ? pd[2] : pt, pr = pd.size() > 3 ? pd[3] : pl;
      int64_t oh = opd.size() > 0 ? opd[0] : 0, ow = opd.size() > 1 ? opd[1] : oh;
      put_f(outn[0], conv_transpose2d(x, w, b, sh, sw, pt, pl, pb, pr, oh, ow, attr_i(node, "group", 1)));
    } else if (op == "Relu") {
      put_f(outn[0], ew::relu(R.f(in[0])));
    } else if (op == "Sigmoid") {
      put_f(outn[0], ew::sigmoid(R.f(in[0])));
    } else if (op == "HardSwish") {
      put_f(outn[0], ew::hardswish(R.f(in[0])));
    } else if (op == "HardSigmoid") {
      put_f(outn[0], ew::hardsigmoid(R.f(in[0]), attr_f(node, "alpha", 0.2f),
                                     attr_f(node, "beta", 0.5f)));
    } else if (op == "Tanh") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::tanh(v); }));
    } else if (op == "Exp") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return fm::exp_(v); }));
    } else if (op == "Log") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::log(v); }));
    } else if (op == "Sqrt") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::sqrt(v); }));
    } else if (op == "Abs") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::fabs(v); }));
    } else if (op == "Clip") {
      float lo = -3.4e38f, hi = 3.4e38f;
      if (in.size() > 1 && !in[1].empty() && R.has_f(in[1])) lo = R.f(in[1])->data[0];
      if (in.size() > 2 && !in[2].empty() && R.has_f(in[2])) hi = R.f(in[2])->data[0];
      lo = attr_f(node, "min", lo); hi = attr_f(node, "max", hi);
      put_f(outn[0], ew::unary(R.f(in[0]), [lo, hi](float v) { return v < lo ? lo : (v > hi ? hi : v); }));
    } else if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow" ||
               op == "Min" || op == "Max") {
      if (R.has_i(in[0]) && R.has_i(in[1])) {          // int64 shape arithmetic
        const IVal& a = R.i(in[0]); const IVal& b = R.i(in[1]);
        IVal o;
        o.dims = a.data.size() >= b.data.size() ? a.dims : b.dims;
        size_t n = a.data.size() > b.data.size() ? a.data.size() : b.data.size();
        o.data.resize(n);
        for (size_t k = 0; k < n; ++k) {
          int64_t x = a.data[a.data.size() == 1 ? 0 : k], y = b.data[b.data.size() == 1 ? 0 : k];
          o.data[k] = op == "Add" ? x + y : op == "Sub" ? x - y : op == "Mul" ? x * y
                    : op == "Div" ? x / y : op == "Min" ? (x < y ? x : y) : (x > y ? x : y);
        }
        R.iv[outn[0]] = o;
      } else {
        Tensor a = R.f(in[0]), b = R.f(in[1]);
        // Commutative ops with a scalar on the left: swap so the fast paths below see (tensor, scalar).
        if ((op == "Add" || op == "Mul") && a->numel() == 1 && b->numel() > 1) std::swap(a, b);
        auto fadd = [](float x, float y) { return x + y; };
        auto fmul = [](float x, float y) { return x * y; };
        auto fsub = [](float x, float y) { return x - y; };
        auto fdiv = [](float x, float y) { return x / y; };
        auto fpow = [](float x, float y) { return std::pow(x, y); };
        auto fmin = [](float x, float y) { return x < y ? x : y; };
        auto fmax = [](float x, float y) { return x > y ? x : y; };
        Tensor y;
        if (op == "Add") {
          y = (b->numel() == 1)         ? ew::add_scalar(a, b->data[0])
            : (a->shape == b->shape)    ? ew::add(a, b)
            : is_channel_bcast(a, b)    ? ew::bcast_channel(a, b, fadd)
            : is_trailing_bcast(a, b)   ? ew::bcast_trailing(a, b, fadd)
                                        : nd::add(a, b);
        } else if (op == "Mul") {
          y = (b->numel() == 1)         ? ew::mul_scalar(a, b->data[0])
            : (a->shape == b->shape)    ? ew::mul(a, b)
            : is_channel_bcast(a, b)    ? ew::bcast_channel(a, b, fmul)
            : is_trailing_bcast(a, b)   ? ew::bcast_trailing(a, b, fmul)
                                        : nd::mul(a, b);
        } else if (op == "Sub") {
          y = (a->shape == b->shape)    ? ew::binary_same(a, b, fsub)
            : is_channel_bcast(a, b)    ? ew::bcast_channel(a, b, fsub)
            : is_trailing_bcast(a, b)   ? ew::bcast_trailing(a, b, fsub)
                                        : nd::sub(a, b);
        } else if (op == "Div") {
          y = (a->shape == b->shape)    ? ew::binary_same(a, b, fdiv)
            : is_channel_bcast(a, b)    ? ew::bcast_channel(a, b, fdiv)
            : is_trailing_bcast(a, b)   ? ew::bcast_trailing(a, b, fdiv)
                                        : nd::div(a, b);
        } else if (op == "Pow") {
          y = (b->numel() == 1) ? ew::unary(a, [e = b->data[0]](float v) { return std::pow(v, e); })
                                : nd::binary(a, b, fpow);
        } else if (op == "Min") {
          y = nd::binary(a, b, fmin);
        } else {
          y = nd::binary(a, b, fmax);
        }
        put_f(outn[0], y);
      }
    } else if (op == "BatchNormalization") {
      put_f(outn[0], ew::batchnorm(R.f(in[0]), R.f(in[1]), R.f(in[2]), R.f(in[3]), R.f(in[4]),
                                   attr_f(node, "epsilon", 1e-5f)));
    } else if (op == "MaxPool") {
      auto ks = attr_ints(node, "kernel_shape"), st = attr_ints(node, "strides"), pd = attr_ints(node, "pads");
      int64_t kh = ks.size() > 0 ? ks[0] : 1, kw = ks.size() > 1 ? ks[1] : kh;
      int64_t sh = st.size() > 0 ? st[0] : 1, sw = st.size() > 1 ? st[1] : sh;
      int64_t pt = pd.size() > 0 ? pd[0] : 0, pl = pd.size() > 1 ? pd[1] : pt;
      if (kh == kw && sh == sw && pt == pl) put_f(outn[0], maxpool2d(R.f(in[0]), kh, sh, pt));
      else { printf("MaxPool: non-square unsupported\n"); std::exit(1); }
    } else if (op == "AveragePool") {
      auto ks = attr_ints(node, "kernel_shape"), st = attr_ints(node, "strides"), pd = attr_ints(node, "pads");
      int64_t kh = ks.size() > 0 ? ks[0] : 1, kw = ks.size() > 1 ? ks[1] : kh;
      int64_t sh = st.size() > 0 ? st[0] : kh, sw = st.size() > 1 ? st[1] : sh;
      int64_t pt = pd.size() > 0 ? pd[0] : 0, pl = pd.size() > 1 ? pd[1] : pt;
      int64_t pb = pd.size() > 2 ? pd[2] : pt, pr = pd.size() > 3 ? pd[3] : pl;
      put_f(outn[0], avgpool2d(R.f(in[0]), kh, kw, sh, sw, pt, pl, pb, pr,
                               attr_i(node, "count_include_pad", 0) != 0,
                               attr_i(node, "ceil_mode", 0) != 0));
    } else if (op == "GlobalAveragePool") {
      put_f(outn[0], ew::gap(R.f(in[0])));
    } else if (op == "Resize") {
      Tensor x = R.f(in[0]);
      int64_t OH = 0, OW = 0;
      if (in.size() > 3 && !in[3].empty() && R.has_i(in[3])) {
        const IVal& s = R.i(in[3]);
        OH = s.data[2]; OW = s.data[3];
      } else if (in.size() > 2 && !in[2].empty() && R.has_f(in[2]) && R.f(in[2])->numel() >= 4) {
        const Tensor sc = R.f(in[2]);
        OH = (int64_t)std::floor(x->shape[2] * (double)sc->data[2]);
        OW = (int64_t)std::floor(x->shape[3] * (double)sc->data[3]);
      } else { OH = x->shape[2] * 2; OW = x->shape[3] * 2; }
      std::string mode = attr_s(node, "mode", "nearest");
      if (mode != "nearest") { printf("Resize: mode '%s' unsupported\n", mode.c_str()); std::exit(1); }
      put_f(outn[0], resize_nearest(x, OH, OW));
    } else if (op == "Concat") {
      int64_t axis = attr_i(node, "axis", 0);
      bool ints = true;
      for (auto& s : in) if (!R.has_i(s)) { ints = false; break; }
      if (ints) {
        IVal o;
        for (auto& s : in) for (int64_t v : R.i(s).data) o.data.push_back(v);
        o.dims = {(int64_t)o.data.size()};
        R.iv[outn[0]] = o;
      } else {
        std::vector<Tensor> xs;
        for (auto& s : in) xs.push_back(R.f(s));
        put_f(outn[0], (axis == 1 && xs[0]->shape.size() == 4) ? concat_ch(xs) : nd::concat(xs, axis));
      }
    } else if (op == "Slice") {
      auto bound = [&](size_t k) {
        return (in.size() > k && !in[k].empty() && R.has_i(in[k])) ? R.i(in[k]).data : std::vector<int64_t>{};
      };
      std::vector<int64_t> st = bound(1), en = bound(2), ax = bound(3), sp = bound(4);
      if (st.empty()) st = attr_ints(node, "starts");
      if (en.empty()) en = attr_ints(node, "ends");
      if (ax.empty()) ax = attr_ints(node, "axes");
      if (R.has_i(in[0])) {                                  // slicing a shape vector
        const IVal& x = R.i(in[0]);
        int64_t n = (int64_t)x.data.size();
        int64_t s = st.empty() ? 0 : st[0], e = en.empty() ? n : en[0], p = sp.empty() ? 1 : sp[0];
        if (s < 0) s += n;
        if (e < 0) e += n;
        s = s < 0 ? 0 : (s > n ? n : s);
        e = e < 0 ? 0 : (e > n ? n : e);
        IVal o;
        for (int64_t k = s; k < e; k += p) o.data.push_back(x.data[(size_t)k]);
        o.dims = {(int64_t)o.data.size()};
        R.iv[outn[0]] = o;
      } else {
        put_f(outn[0], nd::slice(R.f(in[0]), st, en, ax, sp));
      }
    } else if (op == "Shape") {
      Tensor x = R.f(in[0]);
      IVal o;
      o.data = x->shape;
      o.dims = {(int64_t)o.data.size()};
      R.iv[outn[0]] = o;
    } else if (op == "Reshape") {
      Tensor t = R.f(in[0]);
      std::vector<int64_t> shp = R.i(in[1]).data;
      int64_t known = 1; int neg = -1;
      for (size_t k = 0; k < shp.size(); ++k) {
        if (shp[k] == -1) neg = (int)k;
        else if (shp[k] == 0) { shp[k] = t->shape[k]; known *= shp[k]; }   // allowzero=0: copy the dim
        else known *= shp[k];
      }
      if (neg >= 0) shp[(size_t)neg] = t->numel() / known;
      put_f(outn[0], ew::reshape(t, shp));
    } else if (op == "Squeeze" || op == "Unsqueeze") {
      std::vector<int64_t> ax = attr_ints(node, "axes");
      if (ax.empty() && in.size() > 1 && R.has_i(in[1])) ax = R.i(in[1]).data;
      std::vector<int64_t> src = R.has_i(in[0]) ? R.i(in[0]).dims : R.f(in[0])->shape;
      if (R.has_i(in[0]) && src.empty()) src = {(int64_t)R.i(in[0]).data.size()};
      std::vector<int64_t> out;
      if (op == "Squeeze") {
        std::vector<char> drop(src.size(), 0);
        if (ax.empty()) { for (size_t k = 0; k < src.size(); ++k) drop[k] = (char)(src[k] == 1); }
        else for (int64_t a : ax) drop[(size_t)nd::norm_axis(a, src.size())] = 1;
        for (size_t k = 0; k < src.size(); ++k) if (!drop[k]) out.push_back(src[k]);
      } else {
        size_t r = src.size() + ax.size();
        std::vector<char> ins(r, 0);
        for (int64_t a : ax) ins[(size_t)nd::norm_axis(a, r)] = 1;
        size_t si = 0;
        for (size_t k = 0; k < r; ++k) out.push_back(ins[k] ? 1 : src[si++]);
      }
      if (out.empty()) out.push_back(1);
      if (R.has_i(in[0])) { IVal o = R.i(in[0]); o.dims = out; R.iv[outn[0]] = o; }
      else put_f(outn[0], ew::reshape(R.f(in[0]), out));
    } else if (op == "Flatten") {
      Tensor t = R.f(in[0]);
      int64_t axis = attr_i(node, "axis", 1);
      int64_t rows = 1;
      for (int64_t k = 0; k < axis; ++k) rows *= t->shape[(size_t)k];
      put_f(outn[0], ew::reshape(t, {rows, t->numel() / rows}));
    } else if (op == "Transpose") {
      put_f(outn[0], nd::transpose(R.f(in[0]), attr_ints(node, "perm")));
    } else if (op == "MatMul") {
      put_f(outn[0], matmul_nd(R.f(in[0]), R.f(in[1])));
    } else if (op == "Gemm") {
      Tensor a = R.f(in[0]), W = R.f(in[1]);
      Tensor prod = attr_i(node, "transB", 0) ? matmul(a, transpose2d(W)) : matmul(a, W);
      put_f(outn[0], (in.size() >= 3 && !in[2].empty()) ? add_rowvec(prod, R.f(in[2])) : prod);
    } else if (op == "Softmax") {
      Tensor a = R.f(in[0]);
      int64_t axis = attr_i(node, "axis", a->shape.size() == 2 ? 1 : -1);
      bool last = (axis == -1 || axis == (int64_t)a->shape.size() - 1);
      put_f(outn[0], last ? ew::softmax_last(a) : nd::softmax(a, axis));
    } else if (op == "ReduceMean") {
      std::vector<int64_t> ax = attr_ints(node, "axes");
      if (ax.empty() && in.size() > 1 && R.has_i(in[1])) ax = R.i(in[1]).data;
      put_f(outn[0], reduce_mean(R.f(in[0]), ax, attr_i(node, "keepdims", 1) != 0));
    } else if (op == "Gather") {
      int64_t axis = attr_i(node, "axis", 0);
      if (R.has_i(in[0])) {
        const IVal& x = R.i(in[0]); const IVal& ix = R.i(in[1]);
        IVal o;
        for (int64_t k : ix.data) o.data.push_back(x.data[(size_t)(k < 0 ? k + (int64_t)x.data.size() : k)]);
        o.dims = ix.dims;
        R.iv[outn[0]] = o;
      } else {
        Tensor x = R.f(in[0]);
        std::vector<int64_t> idx = R.has_i(in[1]) ? R.i(in[1]).data : std::vector<int64_t>{};
        std::vector<Tensor> rows;
        for (int64_t k : idx) {
          std::vector<int64_t> s(x->shape.size(), 0), e = x->shape;
          s[(size_t)axis] = k < 0 ? k + x->shape[(size_t)axis] : k;
          e[(size_t)axis] = s[(size_t)axis] + 1;
          rows.push_back(nd::slice(x, s, e, {}, {}));
        }
        put_f(outn[0], rows.size() == 1 ? rows[0] : nd::concat(rows, axis));
      }
    } else if (op == "Cast") {
      int64_t to = attr_i(node, "to", 1);
      if (to == 7) {
        if (R.has_i(in[0])) R.iv[outn[0]] = R.i(in[0]);
        else {
          Tensor x = R.f(in[0]);
          IVal o;
          o.dims = x->shape;
          for (float v : x->data) o.data.push_back((int64_t)v);
          R.iv[outn[0]] = o;
        }
      } else if (R.has_i(in[0])) {
        const IVal& x = R.i(in[0]);
        std::vector<float> d;                            // explicit cast: MSVC warns on the iterator ctor
        d.reserve(x.data.size());
        for (int64_t v : x.data) d.push_back((float)v);
        std::vector<int64_t> dims = x.dims.empty() ? std::vector<int64_t>{(int64_t)d.size()} : x.dims;
        put_f(outn[0], from_data(dims, d));
      } else put_f(outn[0], R.f(in[0]));
    } else if (op == "Split") {
      Tensor x = R.f(in[0]);
      int64_t axis = attr_i(node, "axis", 0);
      std::vector<int64_t> parts = attr_ints(node, "split");
      if (parts.empty() && in.size() > 1 && R.has_i(in[1])) parts = R.i(in[1]).data;
      if (parts.empty()) {
        int64_t k = (int64_t)outn.size(), C = x->shape[(size_t)nd::norm_axis(axis, x->shape.size())];
        for (int64_t q = 0; q < k; ++q) parts.push_back(C / k);
      }
      std::vector<Tensor> ps = nd::split(x, axis, parts);
      for (size_t q = 0; q < outn.size() && q < ps.size(); ++q) put_f(outn[q], ps[q]);
    } else if (op == "Identity") {
      if (R.has_i(in[0])) R.iv[outn[0]] = R.i(in[0]);
      else put_f(outn[0], R.f(in[0]));
    } else {
      printf("unsupported ONNX op: %s (node '%s')\n", op.c_str(), node.name.c_str());
      std::exit(1);
    }

    if (prof) {
      double t = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
      prof->add(op, t - tnode);
    }

    if (verbose) {
      double t = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
      auto it = R.fv.find(outn[0]);
      printf("%8.2f ms  %-20s %-30s [", t - tprev, op.c_str(), outn[0].c_str());
      if (it != R.fv.end())
        for (size_t k = 0; k < it->second->shape.size(); ++k)
          printf("%s%lld", k ? "," : "", (long long)it->second->shape[k]);
      printf("]\n");
      fflush(stdout);
      tprev = t;
    }

    // free every input whose last reader was this node
    for (const auto& s : node.input) {
      if (s.empty() || keep.count(s)) continue;
      auto u = uses.find(s);
      if (u == uses.end()) continue;
      if (--u->second <= 0) { R.fv.erase(s); R.iv.erase(s); }
    }
  }

  std::map<std::string, Tensor> out;
  for (const auto& n : keep) {
    auto it = R.fv.find(n);
    if (it != R.fv.end()) out[n] = it->second;
  }
  return out;
}

// A parsed graph plus its materialized weights — what a caller actually wants to hold onto.
struct Model {
  Graph g;
  Weights w;
  std::string in_name, out_name;

  void bind() {
    w = build_weights(g);
    std::set<std::string> initn;
    for (const auto& t : g.init_f) initn.insert(t.name);
    for (const auto& t : g.init_i) initn.insert(t.name);
    for (const auto& n : g.nodes) if (n.op_type == "Constant") initn.insert(n.output[0]);
    in_name.clear();
    for (const auto& vi : g.inputs) if (!initn.count(vi.name)) { in_name = vi.name; break; }
    if (in_name.empty() && !g.inputs.empty()) in_name = g.inputs[0].name;
    out_name = g.outputs.empty() ? std::string() : g.outputs[0].name;
  }
  bool ok() const { return !g.nodes.empty() && !in_name.empty(); }

  // Feed x to the one real input, return the first declared output.
  Tensor run(const Tensor& x, bool verbose = false, Prof* prof = nullptr) const {
    std::vector<std::string> want;
    if (!out_name.empty()) want.push_back(out_name);
    auto res = run_graph(g, {{in_name, x}}, want, verbose, &w, prof);
    if (res.empty()) { printf("Model::run: graph produced no output\n"); std::exit(1); }
    return res.begin()->second;
  }
};

inline Model load_model(const std::string& path) {
  Model m;
  m.g = load_onnx(path);
  m.bind();
  return m;
}
inline Model parse_model(const void* data, size_t n) {
  Model m;
  m.g = parse_onnx(data, n);
  m.bind();
  return m;
}

}  // namespace onx
