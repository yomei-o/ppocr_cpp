// Can the recogniser actually be trained? Overfit one sample and watch the CTC loss fall.
//
//   g++ -std=c++17 -O2 -Ipure -Ipure/third_party -o train_smoke.exe tools/train_smoke.cpp
//   ./train_smoke.exe models/ppocrv5-mobile-rec.onnx models/ppocrv5_dict.txt assets/line_ja.png
//
// WHY OVERFITTING ONE SAMPLE IS THE RIGHT TEST. Checking that gradients are non-zero proves almost
// nothing: a sign error, a transposed weight gradient or a missing accumulation all produce
// perfectly healthy-looking non-zero numbers. Driving one sample's loss down requires the forward
// pass, every backward closure, the accumulator and Adam to agree about what the parameters mean.
// If any one of them disagrees the loss plateaus or diverges instead.
//
// The target is what the model already reads from the image with one character changed. That needs
// no labelled data, keeps the alignment reachable, and still forces real weight movement.
#include "../pure/onnx.hpp"
#include "../pure/onnx_run.hpp"
#include "../pure/ctc.hpp"
#include "../pure/ctc_loss.hpp"
#include "../pure/optim.hpp"
#include "../pure/img.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<uint8_t> read_file(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b((size_t)n);
  if (fread(b.data(), 1, (size_t)n, f) != (size_t)n) { printf("short read\n"); exit(1); }
  fclose(f);
  return b;
}

static im::Img load_bgr(const char* path) {
  int w = 0, h = 0, ch = 0;
  unsigned char* p = stbi_load(path, &w, &h, &ch, 3);
  if (!p) { printf("cannot decode %s\n", path); exit(1); }
  im::Img img = im::make_img(w, h, 3);
  for (int i = 0; i < w * h; ++i) {              // stb gives RGB, the models want BGR
    img.d[(size_t)i * 3 + 0] = p[(size_t)i * 3 + 2];
    img.d[(size_t)i * 3 + 1] = p[(size_t)i * 3 + 1];
    img.d[(size_t)i * 3 + 2] = p[(size_t)i * 3 + 0];
  }
  stbi_image_free(p);
  return img;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);   // so a crash does not swallow the trace
  if (argc < 4) {
    printf("usage: train_smoke <rec.onnx> <dict.txt> <line.png> [steps] [lr]\n");
    return 1;
  }
  const int steps = argc > 4 ? atoi(argv[4]) : 30;
  const float lr = argc > 5 ? (float)atof(argv[5]) : 1e-4f;

  std::vector<uint8_t> buf = read_file(argv[1]);
  onx::Model M = onx::parse_model(buf.data(), buf.size());
  printf("graph: %zu nodes\n", M.g.nodes.size());

  std::vector<uint8_t> dtxt = read_file(argv[2]);
  ctc::Dict dict = ctc::parse_dict(std::string((const char*)dtxt.data(), dtxt.size()));
  printf("dict: %zu entries (index 0 = blank)\n", dict.size());

  Tensor x = im::rec_input(load_bgr(argv[3]));
  printf("input: %lldx%lld\n", (long long)x->shape[3], (long long)x->shape[2]);

  std::vector<Tensor> params = M.trainable();
  size_t nparam = 0;
  for (const auto& p : params) nparam += p->data.size();
  printf("params: %zu tensors, %.2f M values\n", params.size(), nparam / 1e6);

  std::vector<int> target;
  {
    Tensor p0 = M.run(x);
    ctc::Decoded d = ctc::greedy(p0, dict);
    printf("before: \"%s\" (conf %.3f)\n", d.text.c_str(), d.conf);
    const int64_t T = p0->shape[1], C = p0->shape[2];
    int64_t prev = -1;
    for (int64_t t = 0; t < T; ++t) {
      const float* row = p0->data.data() + t * C;
      int64_t best = 0;
      for (int64_t c = 1; c < C; ++c) if (row[c] > row[best]) best = c;
      if (best != 0 && best != prev) target.push_back((int)best);
      prev = best;
    }
    if (target.empty()) { printf("the model reads nothing here; pick another image\n"); return 1; }
    const int old = target[0];
    target[0] = old % ((int)dict.size() - 2) + 1;
    printf("target: %zu ids, first changed %d -> %d\n", target.size(), old, target[0]);
  }

  optim::Adam opt;
  opt.lr = lr;
  float first = 0, last = 0;
  for (int s = 0; s < steps; ++s) {
    Tensor probs = M.run_train(x);
    // The graph ends in Softmax, so log it here and let autograd carry the gradient back through
    // that node. The floor keeps log(0) out of an 18385-wide distribution where most entries are
    // legitimately tiny.
    // The floor is 1e-7, not 1e-12. d/dv log(v) = 1/v, so the floor IS the largest gradient this
    // node can emit: at 1e-12 the very first step produced |g| = 1827 and the second was NaN. The
    // model is saturated (it reads this line at confidence 1.000), so thousands of entries per
    // frame sit at the floor and every one of them multiplies its incoming gradient by 1/eps.
    // 1e-7 still represents "this symbol is not here" — a probability that small never changes a
    // CTC alignment — while bounding the derivative at 1e7 before the clip sees it.
    Tensor lp = gr::log_clamped(probs, 1e-7f);
    Tensor L = ctc::loss_node(lp, target);
    if (!L) {
      printf("step %d: target does not fit in %lld frames\n", s, (long long)probs->shape[1]);
      return 1;
    }
    const float loss = L->data[0];

    backward(L);
    optim::Grads acc;
    acc.add(params);
    const float gn = optim::clip_global_norm(acc, 5.f);
    opt.step(params, acc, 1.f);
    free_graph(L);

    if (s == 0) first = loss;
    last = loss;
    if (s % 5 == 0 || s == steps - 1)
      printf("  step %3d  loss %8.4f   |g| %8.3f\n", s, loss, gn);
  }

  ctc::Decoded after = ctc::greedy(M.run(x), dict);
  printf("after:  \"%s\" (conf %.3f)\n", after.text.c_str(), after.conf);
  printf("loss %.4f -> %.4f  (%.1f%% down)\n", first, last,
         first > 0 ? 100.0 * (first - last) / first : 0.0);
  const bool ok = last < first * 0.5f;
  printf("%s\n", ok ? "train_smoke: OK" : "train_smoke: loss did not halve");
  return ok ? 0 : 1;
}
