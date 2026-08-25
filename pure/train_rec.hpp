// Fine-tuning the recogniser: dataset, label encoding, the loop, and getting the result back out.
//
// LABEL FORMAT is PaddleOCR's: one sample per line, `<image path>\t<text>`. Paths are relative to
// the list file's directory, which is what makes a dataset directory movable.
//
// ENCODING A LABEL is not "look up each character". The dictionary is a list of UTF-8 strings and
// some entries are multi-byte or even multi-codepoint, so the encoder matches longest-first at each
// position. A character the dictionary does not contain is a hard error and not a silent skip: a
// label quietly missing a character trains the model to *not* emit it, which is worse than not
// training at all and impossible to spot from a falling loss.
//
// SAVING. The weights arrived as an ONNX graph and go back out as a side-car file of
// name -> float array, applied over the original model at load. Rewriting the .onnx would mean
// re-encoding protobuf for a 16 MB file to change 4 MB of it, and would lose the property that the
// fine-tune is inspectable and reversible: keep the original, keep the delta.
#pragma once
#include "onnx_run.hpp"
#include "ctc.hpp"
#include "ctc_loss.hpp"
#include "optim.hpp"
#include "img.hpp"
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace trainrec {

struct Sample {
  std::string path;
  std::string text;
};

inline std::string dir_of(const std::string& p) {
  size_t i = p.find_last_of("/\\");
  return i == std::string::npos ? std::string(".") : p.substr(0, i);
}

inline std::vector<Sample> load_list(const std::string& list_path) {
  std::vector<Sample> out;
  FILE* f = fopen(list_path.c_str(), "rb");
  if (!f) { printf("cannot open %s\n", list_path.c_str()); return out; }
  const std::string base = dir_of(list_path);
  std::string line;
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c != '\n') { line.push_back((char)c); continue; }
    while (!line.empty() && line.back() == '\r') line.pop_back();
    size_t tab = line.find('\t');
    if (tab != std::string::npos && tab + 1 < line.size()) {
      std::string p = line.substr(0, tab);
      const bool absolute = p.size() > 1 && (p[0] == '/' || p[1] == ':');
      out.push_back({absolute ? p : base + "/" + p, line.substr(tab + 1)});
    }
    line.clear();
  }
  if (!line.empty()) {
    size_t tab = line.find('\t');
    if (tab != std::string::npos) out.push_back({base + "/" + line.substr(0, tab),
                                                 line.substr(tab + 1)});
  }
  fclose(f);
  return out;
}

// Longest-match encoder. Built once; the dictionary has 18385 entries and a per-character linear
// scan over that for every label would dominate data loading.
struct Encoder {
  std::map<std::string, int> by_text;
  size_t longest = 1;

  explicit Encoder(const ctc::Dict& d) {
    for (size_t i = 1; i < d.ch.size(); ++i) {          // 0 is the blank and has no text
      if (d.ch[i].empty()) continue;
      by_text.emplace(d.ch[i], (int)i);
      if (d.ch[i].size() > longest) longest = d.ch[i].size();
    }
  }

  // Returns false and names the offending byte offset when something is not in the dictionary.
  bool encode(const std::string& s, std::vector<int>& ids, std::string* what = nullptr) const {
    ids.clear();
    size_t i = 0;
    while (i < s.size()) {
      size_t take = 0;
      int id = -1;
      for (size_t n = longest; n >= 1; --n) {
        if (i + n > s.size()) continue;
        auto it = by_text.find(s.substr(i, n));
        if (it != by_text.end()) { take = n; id = it->second; break; }
      }
      if (id < 0) {
        if (what) {
          // report the whole UTF-8 sequence, not one byte, or the message is unreadable
          size_t n = 1;
          const unsigned char b = (unsigned char)s[i];
          if (b >= 0xF0) n = 4; else if (b >= 0xE0) n = 3; else if (b >= 0xC0) n = 2;
          *what = s.substr(i, n);
        }
        return false;
      }
      ids.push_back(id);
      i += take;
    }
    return true;
  }
};

// name -> values, written next to the original .onnx. Magic keeps a stale file from being applied
// to a different model by accident.
inline bool save_weights(const std::string& path, const std::vector<std::string>& names,
                         const std::vector<Tensor>& ts) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { printf("cannot write %s\n", path.c_str()); return false; }
  const char magic[8] = {'P','P','O','C','R','W','1','\0'};
  fwrite(magic, 1, 8, f);
  const uint64_t n = names.size();
  fwrite(&n, sizeof(n), 1, f);
  for (size_t k = 0; k < names.size(); ++k) {
    const uint64_t ln = names[k].size(), nv = ts[k]->data.size();
    fwrite(&ln, sizeof(ln), 1, f);
    fwrite(names[k].data(), 1, ln, f);
    fwrite(&nv, sizeof(nv), 1, f);
    fwrite(ts[k]->data.data(), sizeof(float), nv, f);
  }
  fclose(f);
  return true;
}

inline bool load_weights(const std::string& path, onx::Weights& w, int* applied) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8] = {0};
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PPOCRW1", 7) != 0) {
    printf("%s: not a ppocr weight file\n", path.c_str());
    fclose(f);
    return false;
  }
  uint64_t n = 0;
  if (fread(&n, sizeof(n), 1, f) != 1) { fclose(f); return false; }
  int ok = 0;
  for (uint64_t k = 0; k < n; ++k) {
    uint64_t ln = 0, nv = 0;
    if (fread(&ln, sizeof(ln), 1, f) != 1) break;
    std::string nm(ln, '\0');
    if (fread(&nm[0], 1, ln, f) != ln) break;
    if (fread(&nv, sizeof(nv), 1, f) != 1) break;
    std::vector<float> v((size_t)nv);
    if (fread(v.data(), sizeof(float), (size_t)nv, f) != nv) break;
    auto it = w.f.find(nm);
    // A size mismatch means this file belongs to a different model; skipping it silently would
    // half-apply a fine-tune, so say so.
    if (it == w.f.end()) { printf("  weight '%s' is not in this model, skipped\n", nm.c_str()); continue; }
    if (it->second->data.size() != v.size()) {
      printf("  weight '%s' is %zu values here but %llu in the file, skipped\n", nm.c_str(),
             it->second->data.size(), (unsigned long long)nv);
      continue;
    }
    it->second->data = std::move(v);
    ++ok;
  }
  fclose(f);
  if (applied) *applied = ok;
  return true;
}

struct Config {
  int steps = 500;
  int batch = 4;
  float lr = 1e-4f;
  float clip = 5.f;
  int eval_every = 100;
  int img_h = 48;
  uint32_t seed = 1;
};

// Decoding an image file is the caller's job. img.hpp deliberately does not know about stb — the
// browser build hands it pixels from a canvas and never links an image decoder at all — so the
// loader is passed in rather than assumed.
using LoadFn = im::Img (*)(const std::string&);

// Exact-match accuracy over a list, using greedy decoding — the same thing inference does, so the
// number means what a user would see rather than what the loss says.
struct EvalResult {
  size_t seen = 0, hit = 0, missing = 0;
  float acc() const { return seen ? (float)hit / (float)seen : 0.f; }
};

inline EvalResult evaluate(onx::Model& M, const ctc::Dict& dict, const std::vector<Sample>& data,
                           LoadFn load, int img_h, size_t limit = 0) {
  EvalResult r;
  const size_t n = limit && limit < data.size() ? limit : data.size();
  for (size_t i = 0; i < n; ++i) {
    im::Img crop = load(data[i].path);
    if (crop.empty()) { ++r.missing; continue; }
    Tensor x = im::rec_input(crop, img_h);
    ctc::Decoded d = ctc::greedy(M.run(x), dict);
    if (d.text == data[i].text) ++r.hit;
    ++r.seen;
  }
  return r;
}

// One fine-tuning run. Returns the names of the weights it moved, so the caller can save them.
//
// A "step" is one optimiser update over `batch` samples. Samples are separate forward passes, not
// one padded tensor: the recogniser takes a text line of dynamic width, so padding a batch to the
// widest member would change what the model sees. Gradients are summed across the batch and divided
// by it — see optim.hpp for why the accumulator cannot be skipped.
inline std::vector<std::string> train(onx::Model& M, const ctc::Dict& dict,
                                      const std::vector<Sample>& data,
                                      const std::vector<Sample>& val,
                                      LoadFn load, const Config& cfg) {
  Encoder enc(dict);
  std::vector<Tensor> params = M.trainable();
  std::vector<std::string> names;
  {
    // trainable() returns tensors; recover their names in the same order for saving later.
    std::map<Node*, std::string> byptr;
    for (const auto& kv : M.w.f) if (kv.second) byptr[kv.second.get()] = kv.first;
    for (const auto& p : params) names.push_back(byptr[p.get()]);
  }
  size_t nval = 0;
  for (const auto& p : params) nval += p->data.size();
  printf("training %zu tensors (%.2f M values) on %zu samples\n", params.size(), nval / 1e6,
         data.size());

  optim::Adam opt;
  opt.lr = cfg.lr;
  std::mt19937 rng(cfg.seed);
  std::vector<size_t> order(data.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  size_t cursor = 0;

  std::vector<int> ids;
  std::string bad_char;
  double run_loss = 0;
  int run_n = 0;

  for (int step = 0; step < cfg.steps; ++step) {
    optim::Grads acc;
    int used = 0;
    double batch_loss = 0;
    for (int b = 0; b < cfg.batch; ++b) {
      if (cursor >= order.size()) {
        std::shuffle(order.begin(), order.end(), rng);
        cursor = 0;
      }
      const Sample& sm = data[order[cursor++]];
      im::Img crop = load(sm.path);
      if (crop.empty()) continue;
      if (!enc.encode(sm.text, ids, &bad_char)) {
        printf("  skipping %s: '%s' is not in the dictionary\n", sm.path.c_str(),
               bad_char.c_str());
        continue;
      }
      Tensor x = im::rec_input(crop, cfg.img_h);
      Tensor probs = M.run_train(x);
      Tensor lp = gr::log_clamped(probs, 1e-7f);
      Tensor L = ctc::loss_node(lp, ids);
      if (!L) continue;                       // target longer than the frames available
      backward(L);
      acc.add(params);
      batch_loss += L->data[0];
      ++used;
      free_graph(L);
    }
    if (!used) continue;
    optim::clip_global_norm(acc, cfg.clip);
    opt.step(params, acc, (float)used);
    run_loss += batch_loss / used;
    ++run_n;

    const bool last = (step + 1 == cfg.steps);
    if (cfg.eval_every > 0 && ((step + 1) % cfg.eval_every == 0 || last)) {
      printf("  step %5d  loss %8.4f", step + 1, run_n ? run_loss / run_n : 0.0);
      run_loss = 0;
      run_n = 0;
      if (!val.empty()) {
        EvalResult e = evaluate(M, dict, val, load, cfg.img_h, 200);
        printf("   val exact %.1f%% (%zu/%zu)", 100.0 * e.acc(), e.hit, e.seen);
      }
      printf("\n");
    }
  }
  return names;
}

}  // namespace trainrec
