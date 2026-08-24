// det -> crop -> rec -> CTC, in one call. This is the whole OCR: the two graphs never see each
// other, and everything between them (which quad, which pixels, which resize) is in img.hpp and
// dbnet.hpp.
//
// Each text line is recognised on its own, at its own width (height 48, width = 48 * aspect), not
// padded up to a shared batch width. PaddleOCR batches and pads because it is amortising a Python
// call per batch; here a batch of one costs nothing extra and a crop never carries another crop's
// padding into its own convolutions.
#pragma once
#include "ctc.hpp"
#include "dbnet.hpp"
#include "img.hpp"
#include "onnx_run.hpp"
#include <chrono>
#include <string>
#include <vector>

namespace ppocr {

struct Cfg {
  int det_limit_side_len = 960;
  bool det_limit_max = true;      // 'max': shrink the long side to fit. 'min': grow the short side.
  db::Cfg db;
  int rec_img_h = 48;
  float drop_score = 0.5f;        // discard a line the recognizer is not this sure of
  bool rotate_tall = true;        // read a crop taller than 1.5x its width rotated 90 deg
};

struct Line {
  db::Box box;
  std::string text;
  float conf = 0;
};

struct Result {
  std::vector<Line> lines;
  int det_w = 0, det_h = 0;       // the size the detector actually ran at
  double det_ms = 0, rec_ms = 0;
  int boxes_found = 0;            // before drop_score
};

inline double now_ms() {
  using namespace std::chrono;
  return (double)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// Detection only — the quads, in the source image's pixels, in reading order.
inline std::vector<db::Box> detect(const onx::Model& det, const im::Img& img, const Cfg& cfg,
                                  int* used_w = nullptr, int* used_h = nullptr) {
  im::DetPre pre = im::det_input(img, cfg.det_limit_side_len, cfg.det_limit_max);
  if (used_w) *used_w = pre.rw;
  if (used_h) *used_h = pre.rh;
  Tensor prob = det.run(pre.x);
  std::vector<db::Box> boxes = db::boxes_from_prob(prob, img.w, img.h, cfg.db);
  db::sort_boxes(boxes);
  return boxes;
}

// Recognition of one already-cropped line.
inline ctc::Decoded recognize(const onx::Model& rec, const ctc::Dict& dict, const im::Img& crop,
                              int img_h = 48) {
  Tensor x = im::rec_input(crop, img_h);
  Tensor y = rec.run(x);
  return ctc::greedy(y, dict);
}

inline Result run(const onx::Model& det, const onx::Model& rec, const ctc::Dict& dict,
                  const im::Img& img, const Cfg& cfg = Cfg()) {
  Result out;
  double t0 = now_ms();
  std::vector<db::Box> boxes = detect(det, img, cfg, &out.det_w, &out.det_h);
  out.det_ms = now_ms() - t0;
  out.boxes_found = (int)boxes.size();

  double t1 = now_ms();
  for (const db::Box& b : boxes) {
    im::Img crop = im::crop_quad(img, b.q, cfg.rotate_tall);
    if (crop.w < 2 || crop.h < 2) continue;
    ctc::Decoded d = recognize(rec, dict, crop, cfg.rec_img_h);
    if (d.text.empty() || d.conf < cfg.drop_score) continue;
    Line ln;
    ln.box = b;
    ln.text = d.text;
    ln.conf = d.conf;
    out.lines.push_back(ln);
  }
  out.rec_ms = now_ms() - t1;
  return out;
}

// ---- JSON, the same shape the Python side prints so the two can be diffed ----------------------
inline std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c == '\n') o += "\\n";
    else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); o += buf; }
    else o += (char)c;
  }
  return o;
}

inline std::string to_json(const Result& r) {
  std::string s = "{\"det\":[" + std::to_string(r.det_w) + "," + std::to_string(r.det_h) + "],";
  s += "\"lines\":[";
  for (size_t i = 0; i < r.lines.size(); ++i) {
    const Line& l = r.lines[i];
    char buf[256];
    if (i) s += ",";
    s += "{\"text\":\"" + json_escape(l.text) + "\",";
    snprintf(buf, sizeof buf, "\"conf\":%.4f,\"det_score\":%.4f,\"quad\":[", l.conf, l.box.score);
    s += buf;
    for (int k = 0; k < 4; ++k) {
      snprintf(buf, sizeof buf, "%s[%.1f,%.1f]", k ? "," : "", l.box.q[k].x, l.box.q[k].y);
      s += buf;
    }
    s += "]}";
  }
  s += "]}";
  return s;
}

}  // namespace ppocr
