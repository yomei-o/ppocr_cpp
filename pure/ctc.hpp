// Character table + CTC greedy decode (PaddleOCR's CTCLabelDecode).
//
// The label space is built the way ppocr does it, and the order is not negotiable:
//
//     index 0            = the CTC blank
//     index 1 .. N       = the lines of the dictionary file, in file order
//     index N+1          = a literal space (use_space_char=True)
//
// ppocrv5_dict.txt has 18383 lines and the recognizer's head is 18385 wide, which is exactly
// 1 + 18383 + 1. If those two numbers ever disagree, the table is wrong for the model and every
// character comes out shifted — load_dict() refuses instead of decoding garbage.
#pragma once
#include "autograd.hpp"
#include <cstdio>
#include <string>
#include <vector>

namespace ctc {

struct Dict {
  std::vector<std::string> ch;        // ch[0] is the blank (empty string)
  bool ok() const { return ch.size() > 2; }
  size_t size() const { return ch.size(); }
};

// text is the whole dictionary file. Lines are UTF-8 and kept verbatim (a line may be a space-like
// character such as U+3000, which is why only \r and \n are stripped).
inline Dict parse_dict(const std::string& text) {
  Dict d;
  d.ch.push_back("");                                        // 0 = blank
  size_t i = 0;
  while (i <= text.size()) {
    size_t j = text.find('\n', i);
    if (j == std::string::npos) j = text.size();
    std::string line = text.substr(i, j - i);
    while (!line.empty() && (line.back() == '\r')) line.pop_back();
    if (!(j == text.size() && line.empty())) d.ch.push_back(line);
    if (j == text.size()) break;
    i = j + 1;
  }
  // a trailing newline leaves one empty entry; drop it so the space lands at N+1
  while (d.ch.size() > 1 && d.ch.back().empty()) d.ch.pop_back();
  d.ch.push_back(" ");                                       // use_space_char
  return d;
}

struct Decoded {
  std::string text;
  float conf = 0;
};

// logits is the graph output [1, T, C] — already softmaxed by the graph, so these are
// probabilities. Greedy: argmax per step, collapse repeats, drop the blank, average the kept
// probabilities for the confidence (exactly what CTCLabelDecode reports).
inline Decoded greedy(const Tensor& probs, const Dict& d) {
  Decoded out;
  if (probs->shape.size() != 3) { printf("ctc::greedy: expected [1,T,C]\n"); return out; }
  int64_t T = probs->shape[1], C = probs->shape[2];
  if ((int64_t)d.size() != C) {
    printf("ctc::greedy: dict has %lld entries but the head is %lld wide\n",
           (long long)d.size(), (long long)C);
    return out;
  }
  const float* p = probs->data.data();
  int64_t prev = -1;
  double sum = 0; int64_t n = 0;
  for (int64_t t = 0; t < T; ++t) {
    const float* row = p + t * C;
    int64_t best = 0;
    float bv = row[0];
    for (int64_t c = 1; c < C; ++c) if (row[c] > bv) { bv = row[c]; best = c; }
    if (best != 0 && best != prev) {
      out.text += d.ch[(size_t)best];
      sum += bv;
      ++n;
    }
    prev = best;
  }
  out.conf = n ? (float)(sum / n) : 0.f;
  return out;
}

}  // namespace ctc
