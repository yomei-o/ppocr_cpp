// WASM entry point — the same headers the CLI uses, so the browser runs the same graphs and the
// same decode as `ppocr run`.
//
// Detection and recognition are exposed SEPARATELY on purpose. A page of 50 text lines is one
// detector pass plus 50 recognizer passes, and in WASM that is tens of seconds. Handing the page a
// single blocking `run()` would give it nothing to show for all of it; instead it calls pp_detect
// once, draws the boxes immediately, then calls pp_rec_line(i) in a loop and fills the text in as
// each line lands.
//
// build: sh build/emcc.sh wasm/ppocr_wasm.cpp -o wasm/ppocr.js
#include "pipeline.hpp"
#include <emscripten/emscripten.h>
#include <string>
#include <vector>

static onx::Model g_det, g_rec;
static bool g_det_ok = false, g_rec_ok = false;
static ctc::Dict g_dict;
static im::Img g_img;                       // the frame the boxes belong to
static std::vector<db::Box> g_boxes;
static std::vector<ppocr::Line> g_lines;
static std::string g_json = "{}";
static std::string g_text;
static double g_det_ms = 0, g_rec_ms = 0;

extern "C" {

EMSCRIPTEN_KEEPALIVE int pp_load_det(const unsigned char* buf, int len) {
  g_det = onx::parse_model(buf, (size_t)len);
  g_det_ok = g_det.ok();
  return g_det_ok ? (int)g_det.g.nodes.size() : -1;
}

EMSCRIPTEN_KEEPALIVE int pp_load_rec(const unsigned char* buf, int len) {
  g_rec = onx::parse_model(buf, (size_t)len);
  g_rec_ok = g_rec.ok();
  return g_rec_ok ? (int)g_rec.g.nodes.size() : -1;
}

// The character table. Must match the recognizer head exactly (18385 for PP-OCRv5) — see ctc.hpp.
EMSCRIPTEN_KEEPALIVE int pp_load_dict(const char* text) {
  g_dict = ctc::parse_dict(std::string(text));
  return (int)g_dict.size();
}

EMSCRIPTEN_KEEPALIVE int pp_rec_classes() {
  return g_rec_ok && !g_rec.g.nodes.empty() ? 1 : 0;
}

// Detect on an RGBA frame (w*h*4, canvas order). Returns the box count; the quads are in
// pp_boxes_json() straight away so the page can draw them before any text exists.
EMSCRIPTEN_KEEPALIVE int pp_detect(const unsigned char* rgba, int w, int h, int limit,
                                   float thresh, float box_thresh, float unclip) {
  if (!g_det_ok) return -1;
  g_img = im::from_rgba(rgba, w, h);
  g_boxes.clear();
  g_lines.clear();
  ppocr::Cfg cfg;
  cfg.det_limit_side_len = limit;
  cfg.db.thresh = thresh;
  cfg.db.box_thresh = box_thresh;
  cfg.db.unclip_ratio = unclip;
  double t0 = ppocr::now_ms();
  int uw = 0, uh = 0;
  g_boxes = ppocr::detect(g_det, g_img, cfg, &uw, &uh);
  g_det_ms = ppocr::now_ms() - t0;

  char buf[192];
  g_json = "{\"det\":[" + std::to_string(uw) + "," + std::to_string(uh) + "],\"boxes\":[";
  for (size_t i = 0; i < g_boxes.size(); ++i) {
    if (i) g_json += ",";
    snprintf(buf, sizeof buf, "{\"score\":%.4f,\"quad\":[", g_boxes[i].score);
    g_json += buf;
    for (int k = 0; k < 4; ++k) {
      snprintf(buf, sizeof buf, "%s[%.1f,%.1f]", k ? "," : "", g_boxes[i].q[k].x, g_boxes[i].q[k].y);
      g_json += buf;
    }
    g_json += "]}";
  }
  snprintf(buf, sizeof buf, "],\"ms\":%.0f}", g_det_ms);
  g_json += buf;
  return (int)g_boxes.size();
}

EMSCRIPTEN_KEEPALIVE const char* pp_boxes_json() { return g_json.c_str(); }

// Recognize box i of the frame pp_detect() was last called on. Returns the confidence in
// percent (or -1 on a bad index); the text is in pp_line_text().
EMSCRIPTEN_KEEPALIVE int pp_rec_line(int i, float drop) {
  g_text.clear();
  if (!g_rec_ok || i < 0 || i >= (int)g_boxes.size()) return -1;
  im::Img crop = im::crop_quad(g_img, g_boxes[(size_t)i].q, true);
  if (crop.w < 2 || crop.h < 2) return 0;
  double t0 = ppocr::now_ms();
  ctc::Decoded d = ppocr::recognize(g_rec, g_dict, crop, 48);
  g_rec_ms += ppocr::now_ms() - t0;
  if (d.conf < drop) return 0;                     // kept out of the result, like the CLI does
  g_text = d.text;
  ppocr::Line ln;
  ln.box = g_boxes[(size_t)i];
  ln.text = d.text;
  ln.conf = d.conf;
  g_lines.push_back(ln);
  return (int)(d.conf * 100.f + 0.5f);
}

EMSCRIPTEN_KEEPALIVE const char* pp_line_text() { return g_text.c_str(); }

// Everything recognized so far, in the CLI's JSON shape.
EMSCRIPTEN_KEEPALIVE const char* pp_result() {
  ppocr::Result r;
  r.lines = g_lines;
  r.det_ms = g_det_ms;
  r.rec_ms = g_rec_ms;
  g_json = ppocr::to_json(r);
  return g_json.c_str();
}

// One text line as a rectified crop, for the "what the recognizer actually saw" view. Writes BGR
// bytes into the caller's buffer and returns the packed size (w<<16|h), or 0 if it does not fit.
EMSCRIPTEN_KEEPALIVE int pp_crop(int i, unsigned char* out, int cap) {
  if (i < 0 || i >= (int)g_boxes.size()) return 0;
  im::Img c = im::crop_quad(g_img, g_boxes[(size_t)i].q, true);
  if ((int)c.d.size() > cap) return 0;
  for (size_t k = 0; k + 2 < c.d.size(); k += 3) {   // BGR -> RGB for the canvas
    out[k + 0] = c.d[k + 2];
    out[k + 1] = c.d[k + 1];
    out[k + 2] = c.d[k + 0];
  }
  return (c.w << 16) | c.h;
}

}  // extern "C"
