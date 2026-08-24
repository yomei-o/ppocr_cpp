// Minimal, self-contained ONNX I/O for the pure engine — a hand-rolled protobuf
// codec (no external deps) plus a small in-memory graph IR. Supports the op subset
// yolov8 needs at opset 13: Conv, Sigmoid, Mul, MaxPool, Resize, Concat, Add, Slice.
// This header holds the IR + writer + reader; the interpreter is in onnx_run.hpp.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

namespace onx {

// ---- in-memory graph IR ----
struct Tensor64 { std::string name; std::vector<int64_t> dims; std::vector<float> data; };  // float initializer
struct IntsTensor { std::string name; std::vector<int64_t> dims; std::vector<int64_t> data; }; // int64 initializer
enum AType { A_FLOAT = 1, A_INT = 2, A_STRING = 3, A_FLOATS = 6, A_INTS = 7 };
struct Attr {
  std::string name; int type; int64_t i = 0; float f = 0; std::string s;
  std::vector<int64_t> ints; std::vector<float> floats;
  // AttributeType TENSOR (4): Ultralytics exports pass Reshape/Slice shapes as Constant nodes whose
  // value lives here, not as graph initializers. Skipping it used to make the interpreter read a
  // missing shape and crash.
  bool has_tensor = false; int t_dtype = 1;
  std::vector<int64_t> t_dims, t_ints; std::vector<float> t_floats;
};
struct Node { std::string op_type, name; std::vector<std::string> input, output; std::vector<Attr> attr; };
struct ValueInfo { std::string name; std::vector<int64_t> dims; };  // dim -1 = dynamic
struct Graph {
  std::vector<Node> nodes;
  std::vector<Tensor64> init_f;      // float initializers (weights)
  std::vector<IntsTensor> init_i;    // int64 initializers (slice starts/ends/axes)
  std::vector<ValueInfo> inputs, outputs;
  int opset = 13;
};

// ============================ protobuf writer ============================
struct PB {
  std::string b;
  void varint(uint64_t v) { while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; } b.push_back((char)v); }
  void key(int f, int wt) { varint(((uint64_t)f << 3) | wt); }
  void vint(int f, uint64_t v) { key(f, 0); varint(v); }
  void f32(int f, float x) { key(f, 5); uint32_t u; std::memcpy(&u, &x, 4); for (int i = 0; i < 4; ++i) b.push_back((char)((u >> (8 * i)) & 0xff)); }
  void bytes(int f, const std::string& s) { key(f, 2); varint(s.size()); b += s; }
  void msg(int f, const PB& m) { bytes(f, m.b); }
};

inline std::string raw_of(const std::vector<float>& v) { return std::string((const char*)v.data(), v.size() * 4); }

inline PB build_ftensor(const Tensor64& t) {
  PB p;
  for (int64_t d : t.dims) p.vint(1, (uint64_t)d);   // dims
  p.vint(2, 1);                                       // data_type FLOAT
  p.bytes(8, t.name);                                 // name
  p.bytes(9, raw_of(t.data));                         // raw_data
  return p;
}
inline PB build_itensor(const IntsTensor& t) {
  PB p;
  for (int64_t d : t.dims) p.vint(1, (uint64_t)d);
  p.vint(2, 7);                                       // data_type INT64
  p.bytes(8, t.name);
  std::string raw((const char*)t.data.data(), t.data.size() * 8);
  p.bytes(9, raw);
  return p;
}
inline PB build_attr(const Attr& a) {
  PB p; p.bytes(1, a.name); p.vint(20, a.type);
  switch (a.type) {
    case A_FLOAT:  p.f32(2, a.f); break;
    case A_INT:    p.vint(3, (uint64_t)a.i); break;
    case A_STRING: p.bytes(4, a.s); break;
    case A_INTS:   for (int64_t v : a.ints) p.vint(8, (uint64_t)v); break;
    case A_FLOATS: for (float v : a.floats) p.f32(7, v); break;
  }
  return p;
}
inline PB build_node(const Node& n) {
  PB p;
  for (auto& s : n.input) p.bytes(1, s);
  for (auto& s : n.output) p.bytes(2, s);
  p.bytes(3, n.name); p.bytes(4, n.op_type);
  for (auto& a : n.attr) p.msg(5, build_attr(a));
  return p;
}
inline PB build_valueinfo(const ValueInfo& v) {
  PB shape; for (int64_t d : v.dims) { PB dd; if (d >= 0) dd.vint(1, (uint64_t)d); else dd.bytes(2, "N"); shape.msg(1, dd); }
  PB tt; tt.vint(1, 1); tt.msg(2, shape);            // Tensor: elem_type FLOAT, shape
  PB tp; tp.msg(1, tt);                              // TypeProto.tensor_type
  PB vi; vi.bytes(1, v.name); vi.msg(2, tp);
  return vi;
}

inline void save_onnx(const Graph& g, const std::string& path) {
  PB gp;
  for (auto& n : g.nodes) gp.msg(1, build_node(n));
  gp.bytes(2, "yolov8_cpp");
  for (auto& t : g.init_f) gp.msg(5, build_ftensor(t));
  for (auto& t : g.init_i) gp.msg(5, build_itensor(t));
  for (auto& v : g.inputs) gp.msg(11, build_valueinfo(v));
  for (auto& v : g.outputs) gp.msg(12, build_valueinfo(v));

  PB mp;
  mp.vint(1, 7);                                     // ir_version 7
  mp.bytes(2, "yolov8_cpp");                         // producer_name
  mp.msg(7, gp);                                     // graph
  PB ops; ops.bytes(1, ""); ops.vint(2, (uint64_t)g.opset); mp.msg(8, ops);  // opset_import

  std::ofstream f(path, std::ios::binary); f.write(mp.b.data(), mp.b.size());
}

// ============================ protobuf reader ============================
struct RD {
  const uint8_t* p; const uint8_t* end;
  bool eof() const { return p >= end; }
  uint64_t varint() { uint64_t r = 0; int sh = 0; while (p < end) { uint8_t b = *p++; r |= (uint64_t)(b & 0x7f) << sh; if (!(b & 0x80)) break; sh += 7; } return r; }
  bool tag(int& field, int& wt) { if (p >= end) return false; uint64_t k = varint(); field = (int)(k >> 3); wt = (int)(k & 7); return true; }
  std::string bytes() { uint64_t n = varint(); std::string s((const char*)p, n); p += n; return s; }
  RD sub() { uint64_t n = varint(); RD r{p, p + n}; p += n; return r; }
  float f32() { uint32_t u; std::memcpy(&u, p, 4); p += 4; float f; std::memcpy(&f, &u, 4); return f; }
  void skip(int wt) { if (wt == 0) varint(); else if (wt == 2) { uint64_t n = varint(); p += n; } else if (wt == 5) p += 4; else if (wt == 1) p += 8; }
};

inline void parse_tensor(RD r, Graph& g) {
  std::vector<int64_t> dims; int dtype = 1; std::string name, raw; std::vector<float> fdata; std::vector<int64_t> idata;
  int f, wt;
  while (r.tag(f, wt)) {
    if (f == 1) { if (wt == 2) { RD s = r.sub(); while (!s.eof()) dims.push_back((int64_t)s.varint()); } else dims.push_back((int64_t)r.varint()); }
    else if (f == 2) dtype = (int)r.varint();
    else if (f == 8) name = r.bytes();
    else if (f == 9) raw = r.bytes();
    else if (f == 4) { if (wt == 2) { RD s = r.sub(); while (!s.eof()) fdata.push_back(s.f32()); } else fdata.push_back(r.f32()); }
    else if (f == 7) { if (wt == 2) { RD s = r.sub(); while (!s.eof()) idata.push_back((int64_t)s.varint()); } else idata.push_back((int64_t)r.varint()); }
    else r.skip(wt);
  }
  if (dtype == 7) {                                   // INT64
    IntsTensor t; t.name = name; t.dims = dims;
    if (!raw.empty()) { t.data.resize(raw.size() / 8); std::memcpy(t.data.data(), raw.data(), raw.size()); }
    else t.data = idata;
    g.init_i.push_back(std::move(t));
  } else {                                            // FLOAT
    Tensor64 t; t.name = name; t.dims = dims;
    if (!raw.empty()) { t.data.resize(raw.size() / 4); std::memcpy(t.data.data(), raw.data(), raw.size()); }
    else t.data = fdata;
    g.init_f.push_back(std::move(t));
  }
}

// Parse a TensorProto body into (dims, dtype, ints|floats).
inline void parse_tensor_body(RD r, std::vector<int64_t>& dims, int& dtype,
                             std::vector<int64_t>& ints, std::vector<float>& floats) {
  std::string raw;
  int f, wt;
  dtype = 1;
  while (r.tag(f, wt)) {
    if (f == 1) { if (wt == 2) { RD s2 = r.sub(); while (!s2.eof()) dims.push_back((int64_t)s2.varint()); } else dims.push_back((int64_t)r.varint()); }
    else if (f == 2) dtype = (int)r.varint();
    else if (f == 9) raw = r.bytes();
    else if (f == 4) { if (wt == 2) { RD s2 = r.sub(); while (!s2.eof()) floats.push_back(s2.f32()); } else floats.push_back(r.f32()); }
    else if (f == 7) { if (wt == 2) { RD s2 = r.sub(); while (!s2.eof()) ints.push_back((int64_t)s2.varint()); } else ints.push_back((int64_t)r.varint()); }
    else r.skip(wt);
  }
  if (!raw.empty()) {
    if (dtype == 7) { ints.resize(raw.size() / 8); std::memcpy(ints.data(), raw.data(), raw.size()); }
    else { floats.resize(raw.size() / 4); std::memcpy(floats.data(), raw.data(), raw.size()); }
  }
}

inline Attr parse_attr(RD r) {
  Attr a; int f, wt;
  while (r.tag(f, wt)) {
    if (f == 1) a.name = r.bytes();
    else if (f == 20) a.type = (int)r.varint();
    else if (f == 5) { a.has_tensor = true; parse_tensor_body(r.sub(), a.t_dims, a.t_dtype, a.t_ints, a.t_floats); }
    else if (f == 2) a.f = r.f32();
    else if (f == 3) a.i = (int64_t)r.varint();
    else if (f == 4) a.s = r.bytes();
    else if (f == 7) { if (wt == 2) { RD s = r.sub(); while (!s.eof()) a.floats.push_back(s.f32()); } else a.floats.push_back(r.f32()); }
    else if (f == 8) { if (wt == 2) { RD s = r.sub(); while (!s.eof()) a.ints.push_back((int64_t)s.varint()); } else a.ints.push_back((int64_t)r.varint()); }
    else r.skip(wt);
  }
  return a;
}
inline Node parse_node(RD r) {
  Node n; int f, wt;
  while (r.tag(f, wt)) {
    if (f == 1) n.input.push_back(r.bytes());
    else if (f == 2) n.output.push_back(r.bytes());
    else if (f == 3) n.name = r.bytes();
    else if (f == 4) n.op_type = r.bytes();
    else if (f == 5) n.attr.push_back(parse_attr(r.sub()));
    else r.skip(wt);
  }
  return n;
}
inline std::string parse_valueinfo_name(RD r) {
  int f, wt; std::string name;
  while (r.tag(f, wt)) { if (f == 1) name = r.bytes(); else r.skip(wt); }
  return name;
}

// Full ValueInfoProto: name + static dims (a dynamic dim, e.g. "batch_size", comes back as -1).
// The pipeline reads the detector's input size from here instead of being told it.
inline ValueInfo parse_valueinfo(RD r) {
  ValueInfo vi; int f, wt;
  while (r.tag(f, wt)) {
    if (f == 1) vi.name = r.bytes();
    else if (f == 2) {                                     // TypeProto
      RD tp = r.sub(); int f2, w2;
      while (tp.tag(f2, w2)) {
        if (f2 == 1) {                                     // TypeProto.tensor_type
          RD tt = tp.sub(); int f3, w3;
          while (tt.tag(f3, w3)) {
            if (f3 == 2) {                                 // TensorShapeProto
              RD sh = tt.sub(); int f4, w4;
              while (sh.tag(f4, w4)) {
                if (f4 == 1) {                             // Dimension
                  RD dm = sh.sub(); int f5, w5; int64_t v = -1;
                  while (dm.tag(f5, w5)) {
                    if (f5 == 1) v = (int64_t)dm.varint();  // dim_value
                    else if (f5 == 2) { dm.bytes(); }       // dim_param -> dynamic
                    else dm.skip(w5);
                  }
                  vi.dims.push_back(v);
                } else sh.skip(w4);
              }
            } else tt.skip(w3);
          }
        } else tp.skip(w2);
      }
    } else r.skip(wt);
  }
  return vi;
}
inline void parse_graph(RD r, Graph& g) {
  int f, wt;
  while (r.tag(f, wt)) {
    if (f == 1) g.nodes.push_back(parse_node(r.sub()));
    else if (f == 5) parse_tensor(r.sub(), g);
    else if (f == 11) g.inputs.push_back(parse_valueinfo(r.sub()));
    else if (f == 12) g.outputs.push_back(parse_valueinfo(r.sub()));
    else r.skip(wt);
  }
}
// Parse from memory — the browser fetches the .onnx and hands over the bytes (no filesystem).
inline Graph parse_onnx(const void* data, size_t n) {
  Graph g; RD r{(const uint8_t*)data, (const uint8_t*)data + n};
  int fld, wt;
  while (r.tag(fld, wt)) { if (fld == 7) parse_graph(r.sub(), g); else r.skip(wt); }
  return g;
}

inline Graph load_onnx(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::string buf(n, '\0'); f.read(buf.data(), n);
  return parse_onnx(buf.data(), buf.size());
}

}  // namespace onx
