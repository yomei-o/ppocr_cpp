// Adam, and the gradient accumulator that has to sit in front of it.
//
// WHY AN ACCUMULATOR IS NOT OPTIONAL. autograd.hpp's backward() zeroes the grad of every node it
// can reach before it runs a single closure. Weights are reachable — they are parents on the tape —
// so a second backward() erases what the first one produced. Summing over a batch by calling
// backward() once per sample and hoping the grads add up gives you the last sample's gradient and a
// training curve that looks noisy rather than wrong. Grads are therefore copied out after each
// backward and the optimiser steps from the copy.
//
// That also makes the batch size a property of the training loop rather than of the graph, which
// matters here: PP-OCRv5's recogniser takes one text line of a dynamic width, so "a batch" is
// several separate forward passes, not one padded tensor.
#pragma once
#include "autograd.hpp"
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace optim {

// Sum of gradients over the samples seen since the last clear.
struct Grads {
  std::map<Node*, std::vector<float>> g;

  void add(const std::vector<Tensor>& params) {
    for (const auto& p : params) {
      if (!p || p->grad.empty()) continue;
      auto& acc = g[p.get()];
      if (acc.size() != p->grad.size()) acc.assign(p->grad.size(), 0.f);
      for (size_t i = 0; i < acc.size(); ++i) acc[i] += p->grad[i];
    }
  }
  void clear() { g.clear(); }
  bool empty() const { return g.empty(); }
};

// Adam with bias correction, as in the original paper. No weight decay: fine-tuning a pretrained
// recogniser on a few thousand lines, decay pulls weights toward zero faster than the data pushes
// them anywhere, and the failure is silent — the model gets worse while the loss on the tiny
// fine-tuning set still falls.
struct Adam {
  float lr = 1e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  int64_t t = 0;
  std::map<Node*, std::vector<float>> m, v;

  // scale divides the accumulated gradient — pass the number of samples summed into it.
  void step(const std::vector<Tensor>& params, const Grads& acc, float scale = 1.f) {
    ++t;
    const float bc1 = 1.f - std::pow(b1, (float)t);
    const float bc2 = 1.f - std::pow(b2, (float)t);
    for (const auto& p : params) {
      if (!p) continue;
      auto it = acc.g.find(p.get());
      if (it == acc.g.end()) continue;
      const std::vector<float>& gr = it->second;
      std::vector<float>& mv = m[p.get()];
      std::vector<float>& vv = v[p.get()];
      if (mv.size() != gr.size()) { mv.assign(gr.size(), 0.f); vv.assign(gr.size(), 0.f); }
      const float inv = 1.f / scale;
      for (size_t i = 0; i < gr.size(); ++i) {
        const float gi = gr[i] * inv;
        mv[i] = b1 * mv[i] + (1.f - b1) * gi;
        vv[i] = b2 * vv[i] + (1.f - b2) * gi * gi;
        const float mh = mv[i] / bc1, vh = vv[i] / bc2;
        p->data[i] -= lr * mh / (std::sqrt(vh) + eps);
      }
    }
  }
};

// Clip by global L2 norm. CTC on a long line can produce a very large gradient on the step where
// the alignment first snaps into place; without a clip that one step can undo a whole epoch.
// Returns the norm before clipping, so a caller can see how often it is biting.
inline float clip_global_norm(Grads& acc, float max_norm) {
  double sq = 0;
  for (auto& kv : acc.g) for (float x : kv.second) sq += (double)x * x;
  const float norm = (float)std::sqrt(sq);
  if (norm > max_norm && norm > 0.f) {
    const float s = max_norm / norm;
    for (auto& kv : acc.g) for (float& x : kv.second) x *= s;
  }
  return norm;
}

}  // namespace optim
