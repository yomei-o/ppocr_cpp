// ppocr — PP-OCRv5 text detection + Japanese recognition, from scratch in C++.
//
//   ppocr run  --img photo.jpg [--out result.png] [--json] [--limit 960]
//   ppocr det  --img photo.jpg [--out result.png]
//   ppocr rec  --img line.png                        # one already-cropped text line
//   ppocr info --onnx models/ppocrv5-mobile-det.onnx # op histogram, for porting a new export
//
// The same headers drive the WASM build (wasm/ppocr_wasm.cpp), so the browser runs this decode.
#include "pipeline.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::string buf((size_t)n, '\0');
  f.read(buf.data(), n);
  return buf;
}

static im::Img load_bgr(const std::string& path) {
  int w = 0, h = 0, ch = 0;
  unsigned char* p = stbi_load(path.c_str(), &w, &h, &ch, 3);
  if (!p) { printf("cannot decode %s\n", path.c_str()); std::exit(1); }
  im::Img img = im::make_img(w, h, 3);
  for (int i = 0; i < w * h; ++i) {                      // stb gives RGB, the models want BGR
    img.d[(size_t)i * 3 + 0] = p[(size_t)i * 3 + 2];
    img.d[(size_t)i * 3 + 1] = p[(size_t)i * 3 + 1];
    img.d[(size_t)i * 3 + 2] = p[(size_t)i * 3 + 0];
  }
  stbi_image_free(p);
  return img;
}

static void save_bgr_png(const im::Img& img, const std::string& path) {
  std::vector<unsigned char> rgb((size_t)img.w * img.h * 3);
  for (int i = 0; i < img.w * img.h; ++i) {
    rgb[(size_t)i * 3 + 0] = img.d[(size_t)i * 3 + 2];
    rgb[(size_t)i * 3 + 1] = img.d[(size_t)i * 3 + 1];
    rgb[(size_t)i * 3 + 2] = img.d[(size_t)i * 3 + 0];
  }
  if (!stbi_write_png(path.c_str(), img.w, img.h, 3, rgb.data(), img.w * 3))
    printf("cannot write %s\n", path.c_str());
  else
    printf("wrote %s\n", path.c_str());
}

static const char* arg_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}
static bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], key)) return true;
  return false;
}
static double arg_num(int argc, char** argv, const char* key, double def) {
  const char* s = arg_str(argc, argv, key, nullptr);
  return s ? atof(s) : def;
}

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);                          // so recognised Japanese actually prints
  std::vector<std::string> u8args;
  std::vector<char*> u8argv;
  {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {                                        // main() gets ANSI argv; take the real one
      for (int i = 0; i < wargc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s((size_t)(n > 0 ? n - 1 : 0), '\0');
        if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
        u8args.push_back(s);
      }
      LocalFree(wargv);
      for (std::string& a : u8args) u8argv.push_back(a.data());
      argc = (int)u8argv.size();
      argv = u8argv.data();
    }
  }
#endif

  if (argc < 2) {
    printf("ppocr — PP-OCRv5 detection + Japanese recognition, scratch C++ (no onnxruntime)\n"
           "  ppocr run  --img <file> [--out png] [--json] [--limit 960] [--min] [--thresh 0.3]\n"
           "             [--box-thresh 0.6] [--unclip 1.5] [--drop 0.5] [--det onnx] [--rec onnx]\n"
           "             [--dict txt] [--verbose]\n"
           "  ppocr det  --img <file> [--out png] [--json]\n"
           "  ppocr rec  --img <file>            # one cropped text line\n"
           "  ppocr info --onnx <file>\n");
    return 1;
  }
  std::string cmd = argv[1];
  std::string det_path = arg_str(argc, argv, "--det", "models/ppocrv5-mobile-det.onnx");
  std::string rec_path = arg_str(argc, argv, "--rec", "models/ppocrv5-mobile-rec.onnx");
  std::string dict_path = arg_str(argc, argv, "--dict", "models/ppocrv5_dict.txt");
  const char* img_path = arg_str(argc, argv, "--img", nullptr);
  bool verbose = arg_flag(argc, argv, "--verbose");

  if (cmd == "info") {
    const char* p = arg_str(argc, argv, "--onnx", det_path.c_str());
    onx::Graph g = onx::load_onnx(p);
    std::map<std::string, int> hist;
    for (auto& n : g.nodes) ++hist[n.op_type];
    printf("%s\n  nodes %zu  float-init %zu  int-init %zu  opset %d\n", p, g.nodes.size(),
           g.init_f.size(), g.init_i.size(), g.opset);
    for (auto& kv : hist) printf("    %-22s %d\n", kv.first.c_str(), kv.second);
    for (auto& v : g.inputs) {
      printf("  IN  %s [", v.name.c_str());
      for (size_t i = 0; i < v.dims.size(); ++i) printf("%s%lld", i ? "," : "", (long long)v.dims[i]);
      printf("]\n");
    }
    for (auto& v : g.outputs) {
      printf("  OUT %s [", v.name.c_str());
      for (size_t i = 0; i < v.dims.size(); ++i) printf("%s%lld", i ? "," : "", (long long)v.dims[i]);
      printf("]\n");
    }
    return 0;
  }

  if (!img_path) { printf("--img is required\n"); return 1; }
  im::Img img = load_bgr(img_path);

  ppocr::Cfg cfg;
  cfg.det_limit_side_len = (int)arg_num(argc, argv, "--limit", 960);
  cfg.det_limit_max = !arg_flag(argc, argv, "--min");
  cfg.db.thresh = (float)arg_num(argc, argv, "--thresh", cfg.db.thresh);
  cfg.db.box_thresh = (float)arg_num(argc, argv, "--box-thresh", cfg.db.box_thresh);
  cfg.db.unclip_ratio = (float)arg_num(argc, argv, "--unclip", cfg.db.unclip_ratio);
  cfg.drop_score = (float)arg_num(argc, argv, "--drop", cfg.drop_score);

  if (cmd == "rgba") {
    // Raw canvas-order pixels: what the browser hands the WASM module. wasm/test_node.js reads this
    // so the node smoke test needs no image decoder and sees exactly the CLI's pixels.
    const char* out = arg_str(argc, argv, "--out", "scratch/sample.rgba");
    FILE* f = fopen(out, "wb");
    if (!f) { printf("cannot write %s\n", out); return 1; }
    int w = img.w, h = img.h;
    fwrite(&w, 4, 1, f);
    fwrite(&h, 4, 1, f);
    std::vector<unsigned char> row((size_t)w * 4);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {                      // Img is BGR, canvas wants RGBA
        row[(size_t)x * 4 + 0] = img.at(y, x, 2);
        row[(size_t)x * 4 + 1] = img.at(y, x, 1);
        row[(size_t)x * 4 + 2] = img.at(y, x, 0);
        row[(size_t)x * 4 + 3] = 255;
      }
      fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    printf("wrote %s (%dx%d)\n", out, w, h);
    return 0;
  }

  if (cmd == "dump") {
    // Write the exact input tensor this engine built and the exact output it produced, so
    // tools/parity.py can feed the same input to onnxruntime and diff the two outputs. This is the
    // test that actually covers the interpreter: 664 det nodes and 746 rec nodes against a
    // completely independent implementation of the same graph.
    std::string which = arg_str(argc, argv, "--model", "det");
    const char* out = arg_str(argc, argv, "--out", "scratch/dump.bin");
    onx::Model m = onx::load_model(which == "det" ? det_path : rec_path);
    Tensor x = (which == "det") ? im::det_input(img, cfg.det_limit_side_len, cfg.det_limit_max).x
                                : im::rec_input(img, cfg.rec_img_h);
    Tensor y = m.run(x);
    FILE* f = fopen(out, "wb");
    if (!f) { printf("cannot write %s\n", out); return 1; }
    fwrite("PPCR", 1, 4, f);
    int cnt = 2;
    fwrite(&cnt, 4, 1, f);
    for (const Tensor& t : {x, y}) {
      int nd = (int)t->shape.size();
      fwrite(&nd, 4, 1, f);
      for (int64_t d : t->shape) { int di = (int)d; fwrite(&di, 4, 1, f); }
      fwrite(t->data.data(), 4, (size_t)t->numel(), f);
    }
    fclose(f);
    printf("wrote %s: in [1,3,%lld,%lld] out [", out, (long long)x->shape[2], (long long)x->shape[3]);
    for (size_t i = 0; i < y->shape.size(); ++i) printf("%s%lld", i ? "," : "", (long long)y->shape[i]);
    printf("]\n");
    return 0;
  }

  if (cmd == "bench") {
    // Where the time actually goes. Timing with --verbose does not work: an fflush per node costs
    // more than most nodes do, which makes every op look expensive and the cheap ones look worst.
    int reps = (int)arg_num(argc, argv, "--repeat", 5);
    std::string which = arg_str(argc, argv, "--model", "rec");
    onx::Model m = onx::load_model(which == "det" ? det_path : rec_path);
    Tensor x = (which == "det") ? im::det_input(img, cfg.det_limit_side_len, cfg.det_limit_max).x
                                : im::rec_input(img, cfg.rec_img_h);
    printf("%s  input [1,3,%lld,%lld]  x%d\n", which.c_str(), (long long)x->shape[2],
           (long long)x->shape[3], reps);
    onx::Prof prof;
    double t0 = ppocr::now_ms();
    for (int i = 0; i < reps; ++i) m.run(x, false, &prof);
    double total = ppocr::now_ms() - t0;
    std::vector<std::pair<double, std::string>> rows;
    for (auto& kv : prof.ms) rows.push_back({kv.second, kv.first});
    std::sort(rows.rbegin(), rows.rend());
    printf("%.1f ms / run\n", total / reps);
    for (auto& r : rows)
      printf("  %-22s %8.2f ms/run  (%d calls)\n", r.second.c_str(), r.first / reps,
             prof.n[r.second] / reps);
    return 0;
  }

  if (cmd == "rec") {
    onx::Model rec = onx::load_model(rec_path);
    ctc::Dict dict = ctc::parse_dict(read_file(dict_path));
    printf("dict %zu classes\n", dict.size());
    Tensor x = im::rec_input(img, cfg.rec_img_h);
    printf("input [1,3,%lld,%lld]\n", (long long)x->shape[2], (long long)x->shape[3]);
    Tensor y = rec.run(x, verbose);
    ctc::Decoded d = ctc::greedy(y, dict);
    printf("%s  (conf %.3f, T=%lld)\n", d.text.c_str(), d.conf, (long long)y->shape[1]);
    return 0;
  }

  onx::Model det = onx::load_model(det_path);
  printf("det: %zu nodes\n", det.g.nodes.size());

  if (cmd == "det") {
    int uw = 0, uh = 0;
    double t = ppocr::now_ms();
    std::vector<db::Box> boxes = ppocr::detect(det, img, cfg, &uw, &uh);
    printf("%zu boxes in %.0f ms (%dx%d -> %dx%d)\n", boxes.size(), ppocr::now_ms() - t,
           img.w, img.h, uw, uh);
    for (size_t i = 0; i < boxes.size() && i < 200; ++i)
      printf("  #%2zu score %.3f  (%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f)\n", i,
             boxes[i].score, boxes[i].q[0].x, boxes[i].q[0].y, boxes[i].q[1].x, boxes[i].q[1].y,
             boxes[i].q[2].x, boxes[i].q[2].y, boxes[i].q[3].x, boxes[i].q[3].y);
    const char* out = arg_str(argc, argv, "--out", nullptr);
    if (out) {
      im::Img vis = img;
      for (auto& b : boxes) im::draw_quad(vis, b.q, 60, 220, 60, 2);
      save_bgr_png(vis, out);
    }
    return 0;
  }

  if (cmd != "run") { printf("unknown command '%s'\n", cmd.c_str()); return 1; }

  onx::Model rec = onx::load_model(rec_path);
  ctc::Dict dict = ctc::parse_dict(read_file(dict_path));
  printf("rec: %zu nodes, dict %zu classes\n", rec.g.nodes.size(), dict.size());

  ppocr::Result r = ppocr::run(det, rec, dict, img, cfg);
  if (arg_flag(argc, argv, "--json")) {
    printf("%s\n", ppocr::to_json(r).c_str());
  } else {
    printf("%dx%d -> det %dx%d : %d boxes, %zu lines kept  (det %.0f ms, rec %.0f ms)\n",
           img.w, img.h, r.det_w, r.det_h, r.boxes_found, r.lines.size(), r.det_ms, r.rec_ms);
    for (size_t i = 0; i < r.lines.size(); ++i)
      printf("  %2zu  %.3f  %s\n", i, r.lines[i].conf, r.lines[i].text.c_str());
  }
  const char* out = arg_str(argc, argv, "--out", nullptr);
  if (out) {
    im::Img vis = img;
    for (auto& l : r.lines) im::draw_quad(vis, l.box.q, 60, 220, 60, 2);
    save_bgr_png(vis, out);
  }
  return 0;
}
