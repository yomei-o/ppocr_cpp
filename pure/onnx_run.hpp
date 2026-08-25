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
#include "nd.hpp"          // rank-agnostic transpose/softmax/slice/concat
#include "ew.hpp"          // forward-only elementwise (see that header: the tape ops do not vectorize)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "nn_ops.hpp"      // the tensor ops, lifted out of this file
#include "onnx_grad.hpp"   // gr:: backward wrappers
#include <set>
// Both of the above are included OUTSIDE namespace onx on purpose: onx::Node is a graph node
// (op_type, inputs, outputs) while autograd.hpp's Node is a tensor. Including them inside would
// resolve every `Node*` in the gradient code to the wrong one.

namespace onx {

// Set to true to make run_graph stop at the first node that emits a NaN or an infinity.
inline bool& nan_check() { static bool v = false; return v; }

// ---- an int64 value (a shape, an axes list, a Slice bound) ----
struct IVal { std::vector<int64_t> dims; std::vector<int64_t> data; };

// Drop the tape: no grad buffer, no parents, no backward closure. Without this every intermediate
// stays reachable from its consumers and nothing can be freed early.

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
// TRAINING MODE (train = true) changes exactly three things and nothing else:
//
//   1. infer_only() stays off, so every op allocates a grad buffer and gr::tape records a closure;
//   2. outputs are not passed through nograd(), which would strip that closure again;
//   3. the liveness-based freeing is skipped — an activation whose last *forward* reader has run is
//      still needed by the backward pass, and erasing it is how a tape ends up with dangling holes.
//
// The op dispatch below is shared. A second interpreter for training would be four hundred lines of
// duplicated ONNX decoding that drifts the first time either copy is fixed.
inline std::map<std::string, Tensor> run_graph(const Graph& g,
                                              const std::map<std::string, Tensor>& inputs,
                                              const std::vector<std::string>& want = {},
                                              bool verbose = false,
                                              const Weights* wcache = nullptr,
                                              Prof* prof = nullptr,
                                              bool train = false) {
  infer_only() = !train;                // training needs the grad buffers this would suppress
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

  auto put_f = [&](const std::string& name, const Tensor& t) {
    R.fv[name] = train ? t : nograd(t);
  };
  double tprev = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;

  for (const auto& node : g.nodes) {
    double tnode = prof ? (double)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0
                        : 0.0;
    const std::string& op = node.op_type;
    const auto& in = node.input;
    const auto& outn = node.output;

    // Refuse to train through an op with no backward rather than produce a zero gradient.
    //
    // An unwrapped op is not an error at runtime: it returns a value with no tape entry, backward()
    // walks past it, and every parameter behind it receives exactly zero. The run completes, the
    // loss even falls — because the layers the tape *can* reach still learn — and the rest of the
    // model silently never moves. Naming the differentiable set here turns that into a stop.
    //
    // The ops listed without a gr:: wrapper are the ones that carry no float gradient by
    // construction: Constant and Shape produce values, Cast/Gather/Range/Squeeze on the int64 shape
    // stream are index arithmetic, and Identity is a rename.
    if (train) {
      static const std::set<std::string> kDifferentiable = {
        "Conv", "Relu", "Sigmoid", "HardSwish", "HardSigmoid", "Sqrt",
        "Add", "Sub", "Mul", "Div", "Pow", "BatchNormalization", "AveragePool",
        "GlobalAveragePool", "Concat", "Slice", "Reshape", "Squeeze", "Unsqueeze",
        "Transpose", "MatMul", "Softmax", "ReduceMean", "Flatten", "Gemm",
        // no float gradient by construction
        "Constant", "Shape", "Cast", "Gather", "Identity", "Range", "Expand", "ConstantOfShape",
      };
      if (!kDifferentiable.count(op)) {
        fprintf(stderr,
                "onnx_run: training requested but '%s' has no backward in onnx_grad.hpp.\n"
                "  Every parameter behind this node would get a gradient of zero and the run\n"
                "  would look like it was learning. Add a gr:: wrapper for it, or keep this\n"
                "  graph on the inference path.\n", op.c_str());
        std::exit(2);
      }
    }

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
      if (train)
        put_f(outn[0], depthwise ? gr::dwconv2d(x, w, b, sh, sw, pt, pl, pb, pr)
                                 : gr::conv2d(x, w, b, sh, sw, pt, pl, pb, pr, grp));
      else
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
      put_f(outn[0], train ? gr::relu(R.f(in[0])) : ew::relu(R.f(in[0])));
    } else if (op == "Sigmoid") {
      put_f(outn[0], train ? gr::sigmoid(R.f(in[0])) : ew::sigmoid(R.f(in[0])));
    } else if (op == "HardSwish") {
      put_f(outn[0], train ? gr::hardswish(R.f(in[0])) : ew::hardswish(R.f(in[0])));
    } else if (op == "HardSigmoid") {
      { const float al = attr_f(node, "alpha", 0.2f), be = attr_f(node, "beta", 0.5f);
        put_f(outn[0], train ? gr::hardsigmoid(R.f(in[0]), al, be)
                             : ew::hardsigmoid(R.f(in[0]), al, be)); }
    } else if (op == "Tanh") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::tanh(v); }));
    } else if (op == "Exp") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return fm::exp_(v); }));
    } else if (op == "Log") {
      put_f(outn[0], ew::unary(R.f(in[0]), [](float v) { return std::log(v); }));
    } else if (op == "Sqrt") {
      put_f(outn[0], train ? gr::sqrt(R.f(in[0]))
                           : ew::unary(R.f(in[0]), [](float v) { return std::sqrt(v); }));
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
        if (train) {
          // One general path when training. The four fast paths below (scalar, same-shape, channel
          // broadcast, trailing broadcast) exist because they measured faster on these graphs, and
          // none of them builds a tape — routing through them during training is exactly how the
          // first end-to-end run produced a gradient norm of precisely zero while every op was
          // nominally "supported". gr::* handles all the broadcast shapes with one implementation.
          y = op == "Add" ? gr::add(a, b)
            : op == "Sub" ? gr::sub(a, b)
            : op == "Mul" ? gr::mul(a, b)
            : op == "Div" ? gr::div(a, b)
            : op == "Pow" ? gr::pow(a, b)
                          : nullptr;
          if (!y) {
            fprintf(stderr, "onnx_run: training through '%s' is not supported\n", op.c_str());
            std::exit(2);
          }
        } else if (op == "Add") {
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
      { const float eps = attr_f(node, "epsilon", 1e-5f);
        put_f(outn[0], train ? gr::batchnorm(R.f(in[0]), R.f(in[1]), R.f(in[2]), R.f(in[3]),
                                             R.f(in[4]), eps)
                             : ew::batchnorm(R.f(in[0]), R.f(in[1]), R.f(in[2]), R.f(in[3]),
                                             R.f(in[4]), eps)); }
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
      { const bool cip = attr_i(node, "count_include_pad", 0) != 0;
        const bool cm = attr_i(node, "ceil_mode", 0) != 0;
        put_f(outn[0], train ? gr::avgpool(R.f(in[0]), kh, kw, sh, sw, pt, pl, pb, pr, cip, cm)
                             : avgpool2d(R.f(in[0]), kh, kw, sh, sw, pt, pl, pb, pr, cip, cm)); }
    } else if (op == "GlobalAveragePool") {
      put_f(outn[0], train ? gr::gap(R.f(in[0])) : ew::gap(R.f(in[0])));
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
        put_f(outn[0], train ? gr::concat(xs, axis)
                           : ((axis == 1 && xs[0]->shape.size() == 4) ? concat_ch(xs)
                                                                     : nd::concat(xs, axis)));
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
      put_f(outn[0], train ? gr::reshape(t, shp) : ew::reshape(t, shp));
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
      else put_f(outn[0], train ? gr::reshape(R.f(in[0]), out) : ew::reshape(R.f(in[0]), out));
    } else if (op == "Flatten") {
      Tensor t = R.f(in[0]);
      int64_t axis = attr_i(node, "axis", 1);
      int64_t rows = 1;
      for (int64_t k = 0; k < axis; ++k) rows *= t->shape[(size_t)k];
      { const std::vector<int64_t> fs = {rows, t->numel() / rows};
        put_f(outn[0], train ? gr::reshape(t, fs) : ew::reshape(t, fs)); }
    } else if (op == "Transpose") {
      { const auto pm = attr_ints(node, "perm");
        put_f(outn[0], train ? gr::transpose(R.f(in[0]), pm) : nd::transpose(R.f(in[0]), pm)); }
    } else if (op == "MatMul") {
      put_f(outn[0], train ? gr::matmul(R.f(in[0]), R.f(in[1]))
                           : matmul_nd(R.f(in[0]), R.f(in[1])));
    } else if (op == "Gemm") {
      Tensor a = R.f(in[0]), W = R.f(in[1]);
      const bool tb = attr_i(node, "transB", 0) != 0;
      Tensor B = tb ? (train ? gr::transpose(W, {1, 0}) : nd::transpose(W, {1, 0})) : W;
      Tensor prod = train ? gr::matmul(a, B) : matmul_nd(a, B);
      const bool has_bias = in.size() >= 3 && !in[2].empty();
      put_f(outn[0], !has_bias ? prod
                     : train   ? gr::add(prod, R.f(in[2]))
                               : ew::bcast_trailing(prod, R.f(in[2]),
                                                    [](float x, float y) { return x + y; }));
    } else if (op == "Softmax") {
      Tensor a = R.f(in[0]);
      int64_t axis = attr_i(node, "axis", a->shape.size() == 2 ? 1 : -1);
      bool last = (axis == -1 || axis == (int64_t)a->shape.size() - 1);
      put_f(outn[0], train ? gr::softmax(a, last ? -1 : axis)
                           : (last ? ew::softmax_last(a) : nd::softmax(a, axis)));
    } else if (op == "ReduceMean") {
      std::vector<int64_t> ax = attr_ints(node, "axes");
      if (ax.empty() && in.size() > 1 && R.has_i(in[1])) ax = R.i(in[1]).data;
      { const bool kd = attr_i(node, "keepdims", 1) != 0;
        put_f(outn[0], train ? gr::reduce_mean(R.f(in[0]), ax, kd)
                             : reduce_mean(R.f(in[0]), ax, kd)); }
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

    // Optional: name the first node whose output goes non-finite. Off by default (one pass over
    // every activation is not free); the training tools turn it on when a loss goes NaN.
    if (nan_check() && !outn.empty()) {
      auto bad = [](const Tensor& t) {
        if (!t) return false;
        for (float v : t->data) if (!(v == v) || v > 3.4e38f || v < -3.4e38f) return true;
        return false;
      };
      // A tensor whose data vector is shorter than its shape says is worse than a NaN: the ops
      // size their loops from the shape, so the next one reads past the end and gets whatever is
      // in the allocator. That shows up downstream as a partially non-finite result with no
      // arithmetic explanation, which is exactly what sent this hunt to Pow.
      for (const auto& nm : outn) {
        if (nm.empty()) continue;
        auto o = R.fv.find(nm);
        if (o == R.fv.end() || !o->second) continue;
        int64_t want_n = 1;
        for (int64_t d : o->second->shape) want_n *= d;
        if ((int64_t)o->second->data.size() != want_n) {
          fprintf(stderr, "onnx_run: '%s' (%s) shape says %lld values but data holds %zu\n",
                  op.c_str(), nm.c_str(), (long long)want_n, o->second->data.size());
          std::exit(3);
        }
      }
      auto it = R.fv.find(outn[0]);
      if (it != R.fv.end() && bad(it->second)) {
        {
          const Tensor& ot = it->second;
          size_t nbad = 0;
          float omax = 0;
          for (float v : ot->data) {
            if (!(v == v) || v > 3.4e38f || v < -3.4e38f) ++nbad;
            else { const float av = v < 0 ? -v : v; if (av > omax) omax = av; }
          }
          fprintf(stderr, "onnx_run: '%s' (%s) out shape[", op.c_str(), outn[0].c_str());
          for (size_t k = 0; k < ot->shape.size(); ++k)
            fprintf(stderr, "%s%lld", k ? "," : "", (long long)ot->shape[k]);
          fprintf(stderr, "] n=%zu bad=%zu finite-max=%.4g; inputs:", ot->data.size(), nbad, omax);
        }
        auto maxabs = [](const Tensor& t) {
          float m = 0;
          if (t) for (float v : t->data) { const float a = v < 0 ? -v : v; if (a > m) m = a; }
          return m;
        };
        for (const auto& nm : in) {
          if (nm.empty()) continue;
          auto f = R.fv.find(nm);
          if (f == R.fv.end()) { fprintf(stderr, " %s=int", nm.c_str()); continue; }
          fprintf(stderr, " %s=%s(max|v|=%.4g)", nm.c_str(), bad(f->second) ? "BAD" : "ok",
                  maxabs(f->second));
        }
        fprintf(stderr, "\n");
        {
          const Tensor& ot = it->second;
          const Tensor in0 = R.fv.count(in[0]) ? R.fv[in[0]] : nullptr;
          int shown = 0;
          for (size_t k = 0; k < ot->data.size() && shown < 5; ++k) {
            const float v = ot->data[k];
            if (v == v && v <= 3.4e38f && v >= -3.4e38f) continue;
            fprintf(stderr, "   bad[%zu] = %g   in0[%zu] = %g\n", k, v, k,
                    (in0 && k < in0->data.size()) ? in0->data[k] : 0.f);
            ++shown;
          }
        }
        std::exit(3);
      }
    }

    // free every input whose last reader was this node — forward only. Under train the tape
    // holds these as parents and the backward pass reads their data again.
    if (train) continue;
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

  // Same graph, with a tape. The returned tensor is the root of it: call backward() on a loss built
  // from it, then free_graph(). Not const — the weights in w accumulate gradients.
  Tensor run_train(const Tensor& x) {
    // Every float weight needs somewhere for a gradient to land, whether or not the optimiser will
    // ever use it. These are two different questions and conflating them segfaults: the Pow
    // exponent is not a parameter, but it IS a parent on the tape, and gr::pow's backward writes
    // into its grad. trainable() decides what Adam touches; this decides what backward can write.
    for (auto& kv : w.f) {
      Tensor& t = kv.second;
      if (t && t->grad.size() != t->data.size()) t->grad.assign(t->data.size(), 0.f);
    }
    std::vector<std::string> want;
    if (!out_name.empty()) want.push_back(out_name);
    auto res = run_graph(g, {{in_name, x}}, want, false, &w, nullptr, true);
    if (res.empty()) { printf("Model::run_train: graph produced no output\n"); std::exit(1); }
    return res.begin()->second;
  }

  // The weights this graph can learn.
  //
  // NOT "every float constant". paddle2onnx emits weights as Constant nodes, but it emits other
  // things that way too, and one of them will destroy the model on the first optimiser step:
  //
  //     Pow(x - mean, 2.0)          <- the 2.0 is a Constant, i.e. a float tensor in w.f
  //
  // Train that and Adam nudges the exponent to 2.0003. pow() of a NEGATIVE base to a non-integer
  // power is NaN, so every below-mean activation in that LayerNorm turns to NaN on step 1 while the
  // above-mean ones stay finite. The symptom is a partially non-finite tensor with no arithmetic
  // explanation; the cause is calling a constant a parameter. Clip bounds, epsilons and scale
  // factors are the same story with quieter failures.
  //
  // So a tensor is trainable only where the op it feeds says it is a parameter: a convolution's
  // kernel or bias, a matmul's right-hand matrix, a normalisation's scale and shift. Anything a
  // graph merely happens to store as floats stays frozen.
  //
  // BatchNorm's inputs 3 and 4 (running mean and variance) are deliberately excluded — see the note
  // in onnx_grad.hpp on why the statistics stay frozen during fine-tuning.
  std::vector<Tensor> trainable() {
    std::set<std::string> want;
    auto take = [&](const Node& n, size_t i) {
      if (i < n.input.size() && !n.input[i].empty() && w.f.count(n.input[i])) want.insert(n.input[i]);
    };
    for (const auto& n : g.nodes) {
      if (n.op_type == "Conv" || n.op_type == "ConvTranspose") { take(n, 1); take(n, 2); }
      else if (n.op_type == "Gemm")                            { take(n, 1); take(n, 2); }
      else if (n.op_type == "MatMul")                          { take(n, 1); }
      else if (n.op_type == "BatchNormalization")              { take(n, 1); take(n, 2); }
      else if (n.op_type == "Add" || n.op_type == "Mul") {
        // A folded bias or per-channel scale: a constant shaped like [C] or [1,C,1,1]. These are
        // real parameters — paddle2onnx turns conv biases into exactly this. A scalar is not: that
        // is a multiplier baked into the graph, and it is where the Pow exponent problem lives.
        for (size_t i = 0; i < n.input.size(); ++i) {
          auto it = w.f.find(n.input[i]);
          if (it == w.f.end() || !it->second) continue;
          const Tensor& t = it->second;
          if (t->data.size() <= 1) continue;
          int64_t nonone = 0;
          for (int64_t d : t->shape) if (d != 1) ++nonone;
          if (nonone == 1) want.insert(n.input[i]);
        }
      }
    }
    std::vector<Tensor> ps;
    for (const auto& nm : want) {
      Tensor& t = w.f[nm];
      if (!t || t->data.empty()) continue;
      if (t->grad.size() != t->data.size()) t->grad.assign(t->data.size(), 0.f);
      t->requires_grad = true;
      ps.push_back(t);
    }
    return ps;
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
