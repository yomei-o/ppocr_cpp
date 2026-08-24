// DB (Differentiable Binarization) post-processing — the half of text detection that is not the
// network. The graph hands back one probability map; everything that turns it into rotated quads
// lives here, transcribed from PaddleOCR's DBPostProcess (ppocr/postprocess/db_postprocess.py).
//
// The pipeline per connected component of `prob > thresh`:
//   trace the outer contour  ->  Douglas-Peucker at eps = 0.002 * arcLength
//   -> score = mean prob inside that polygon (reject below box_thresh)
//   -> unclip: grow by d = area * unclip_ratio / perimeter
//   -> min-area rectangle -> 4 points ordered top-left, top-right, bottom-right, bottom-left
//
// UNCLIP IS DONE ON THE RECTANGLE, NOT THE POLYGON. PaddleOCR offsets the polygon with pyclipper
// (a round join, i.e. a Minkowski sum with a disc of radius d) and only then takes the min-area
// rectangle. Growing the min-area rectangle by d on both axes gives the same rectangle for any
// blob whose optimal orientation the offset does not change — which is every text line — and costs
// a few lines instead of a polygon-clipping library. The d itself still comes from the traced
// polygon's own area and perimeter, so a jagged contour still unclips less than a smooth one.
#pragma once
#include "autograd.hpp"
#include "img.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace db {

using im::Pt;

struct Cfg {
  float thresh = 0.3f;          // binarize the probability map here
  float box_thresh = 0.6f;      // drop a candidate whose mean probability is below this
  float unclip_ratio = 1.5f;
  int max_candidates = 1000;
  int min_size = 3;
};

struct Box {
  Pt q[4];                      // top-left, top-right, bottom-right, bottom-left
  float score = 0;
};

// ---- small geometry ---------------------------------------------------------------------------
inline double poly_area(const std::vector<Pt>& p) {
  double a = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const Pt& u = p[i];
    const Pt& v = p[(i + 1) % p.size()];
    a += u.x * v.y - v.x * u.y;
  }
  return std::fabs(a) * 0.5;
}
inline double poly_len(const std::vector<Pt>& p) {
  double L = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const Pt& u = p[i];
    const Pt& v = p[(i + 1) % p.size()];
    L += std::hypot(v.x - u.x, v.y - u.y);
  }
  return L;
}

// Douglas-Peucker on a closed contour, matching cv2.approxPolyDP(closed=True) closely enough for
// the epsilon PaddleOCR uses (0.002 * arcLength — a very light simplification).
inline void dp_rec(const std::vector<Pt>& p, int i0, int i1, double eps, std::vector<char>& keep) {
  if (i1 <= i0 + 1) return;
  double ax = p[i0].x, ay = p[i0].y, bx = p[i1].x, by = p[i1].y;
  double dx = bx - ax, dy = by - ay, nrm = std::hypot(dx, dy);
  int worst = -1; double wd = -1;
  for (int i = i0 + 1; i < i1; ++i) {
    double d = nrm < 1e-12 ? std::hypot(p[i].x - ax, p[i].y - ay)
                           : std::fabs(dy * (p[i].x - ax) - dx * (p[i].y - ay)) / nrm;
    if (d > wd) { wd = d; worst = i; }
  }
  if (wd <= eps || worst < 0) return;
  keep[(size_t)worst] = 1;
  dp_rec(p, i0, worst, eps, keep);
  dp_rec(p, worst, i1, eps, keep);
}

inline std::vector<Pt> approx_poly(const std::vector<Pt>& c, double eps) {
  if (c.size() < 4) return c;
  // split the closed ring at the two farthest-apart-ish anchors, then simplify both arcs
  size_t n = c.size();
  size_t far = 0; double best = -1;
  for (size_t i = 1; i < n; ++i) {
    double d = std::hypot(c[i].x - c[0].x, c[i].y - c[0].y);
    if (d > best) { best = d; far = i; }
  }
  std::vector<char> keep(n, 0);
  keep[0] = 1; keep[far] = 1;
  dp_rec(c, 0, (int)far, eps, keep);
  std::vector<Pt> tail(c.begin() + (long)far, c.end());
  tail.push_back(c[0]);
  std::vector<char> keep2(tail.size(), 0);
  keep2[0] = 1; keep2[tail.size() - 1] = 1;
  dp_rec(tail, 0, (int)tail.size() - 1, eps, keep2);
  std::vector<Pt> out;
  for (size_t i = 0; i < n; ++i) if (keep[i] && i <= far) out.push_back(c[i]);
  for (size_t i = 1; i + 1 < tail.size(); ++i) if (keep2[i]) out.push_back(tail[i]);
  return out;
}

// Andrew monotone chain.
inline std::vector<Pt> convex_hull(std::vector<Pt> p) {
  if (p.size() < 3) return p;
  std::sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  auto cross = [](const Pt& o, const Pt& a, const Pt& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  };
  std::vector<Pt> h(p.size() * 2);
  size_t k = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
    h[k++] = p[i];
  }
  size_t lower = k + 1;
  for (size_t i = p.size() - 1; i-- > 0;) {
    while (k >= lower && cross(h[k - 2], h[k - 1], p[i]) <= 0) --k;
    h[k++] = p[i];
  }
  h.resize(k - 1);
  return h;
}

// Rotating calipers: the minimum-area enclosing rectangle, returned as centre + two axes.
struct MinRect { Pt c; double ux = 1, uy = 0; double a = 0, b = 0; };   // half-extents a (along u), b

// `grow` is the unclip distance, and it is part of the *objective*, not applied afterwards.
// PaddleOCR offsets the polygon first and only then asks for the minimum-area rectangle, so the
// orientation it picks minimises (w + 2d)(h + 2d), not w*h — those differ, and picking the
// ungrown optimum is what left this a pixel or two off the reference on near-square blobs.
// (Everything else about the two is identical: a round polygon offset is a Minkowski sum with a
// disc of radius d, and for a fixed orientation the enclosing rectangle of that sum is exactly the
// original rectangle grown by d on both axes.)
inline MinRect min_area_rect(const std::vector<Pt>& pts, double grow = 0.0) {
  MinRect best;
  std::vector<Pt> h = convex_hull(pts);
  if (h.size() < 2) {
    if (!h.empty()) best.c = h[0];
    return best;
  }
  double bestArea = 1e300;
  for (size_t i = 0; i < h.size(); ++i) {
    const Pt& p0 = h[i];
    const Pt& p1 = h[(i + 1) % h.size()];
    double ex = p1.x - p0.x, ey = p1.y - p0.y, L = std::hypot(ex, ey);
    if (L < 1e-9) continue;
    ex /= L; ey /= L;
    double u0 = 1e300, u1 = -1e300, v0 = 1e300, v1 = -1e300;
    for (const Pt& q : h) {
      double u = q.x * ex + q.y * ey;
      double v = -q.x * ey + q.y * ex;
      u0 = std::min(u0, u); u1 = std::max(u1, u);
      v0 = std::min(v0, v); v1 = std::max(v1, v);
    }
    double area = (u1 - u0 + 2 * grow) * (v1 - v0 + 2 * grow);
    if (area < bestArea) {
      bestArea = area;
      double uc = (u0 + u1) * 0.5, vc = (v0 + v1) * 0.5;
      best.ux = ex; best.uy = ey;
      best.c.x = uc * ex - vc * ey;
      best.c.y = uc * ey + vc * ex;
      best.a = (u1 - u0) * 0.5;
      best.b = (v1 - v0) * 0.5;
    }
  }
  return best;
}

// get_mini_boxes' ordering: sort by x, then pick top/bottom within each pair.
inline void order_quad(Pt in[4], Pt out[4]) {
  Pt p[4] = {in[0], in[1], in[2], in[3]};
  std::sort(p, p + 4, [](const Pt& a, const Pt& b) { return a.x < b.x; });
  int i1, i2, i3, i4;
  if (p[1].y > p[0].y) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
  if (p[3].y > p[2].y) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
  out[0] = p[i1]; out[1] = p[i2]; out[2] = p[i3]; out[3] = p[i4];
}

inline void rect_corners(const MinRect& r, double grow, Pt out[4]) {
  double a = r.a + grow, b = r.b + grow;
  double vx = -r.uy, vy = r.ux;
  Pt c[4];
  c[0] = {r.c.x - a * r.ux - b * vx, r.c.y - a * r.uy - b * vy};
  c[1] = {r.c.x + a * r.ux - b * vx, r.c.y + a * r.uy - b * vy};
  c[2] = {r.c.x + a * r.ux + b * vx, r.c.y + a * r.uy + b * vy};
  c[3] = {r.c.x - a * r.ux + b * vx, r.c.y - a * r.uy + b * vy};
  order_quad(c, out);
}

// mean of `prob` over the pixels inside `poly`, restricted to the polygon's bounding box
// (box_score_fast).
inline float poly_score(const float* prob, int W, int H, const std::vector<Pt>& poly) {
  double xmn = 1e300, xmx = -1e300, ymn = 1e300, ymx = -1e300;
  for (const Pt& p : poly) {
    xmn = std::min(xmn, p.x); xmx = std::max(xmx, p.x);
    ymn = std::min(ymn, p.y); ymx = std::max(ymx, p.y);
  }
  int x0 = std::min(std::max((int)std::floor(xmn), 0), W - 1);
  int x1 = std::min(std::max((int)std::ceil(xmx), 0), W - 1);
  int y0 = std::min(std::max((int)std::floor(ymn), 0), H - 1);
  int y1 = std::min(std::max((int)std::ceil(ymx), 0), H - 1);
  double sum = 0; int64_t cnt = 0;
  size_t n = poly.size();
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      bool inside = false;                                  // even-odd crossing at the pixel centre
      for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double yi = poly[i].y, yj = poly[j].y;
        if ((yi > y) != (yj > y)) {
          double t = (y - yi) / (yj - yi);
          if (poly[i].x + t * (poly[j].x - poly[i].x) > x) inside = !inside;
        }
      }
      if (inside) { sum += prob[(size_t)y * W + x]; ++cnt; }
    }
  return cnt ? (float)(sum / cnt) : 0.f;
}

// ---- outer-contour tracing -------------------------------------------------------------------
// Moore-neighbour tracing of one 8-connected component's outer boundary, plus a flood fill so the
// component is visited once. cv2.findContours(RETR_LIST) would also return hole contours; those
// only ever produce candidates that box_thresh throws away, so they are skipped here.
inline std::vector<std::vector<Pt>> trace_contours(const std::vector<unsigned char>& bin, int W, int H) {
  std::vector<char> seen((size_t)W * H, 0);
  std::vector<std::vector<Pt>> out;
  const int dx8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  auto fg = [&](int x, int y) { return x >= 0 && y >= 0 && x < W && y < H && bin[(size_t)y * W + x]; };

  std::vector<int> stack;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      size_t idx = (size_t)y * W + x;
      if (!bin[idx] || seen[idx]) continue;

      // trace from this pixel (the first foreground pixel of the component in scan order, so the
      // background is to its left/up: start the search there)
      std::vector<Pt> contour;
      int cx = x, cy = y, bdir = 4;                          // came from the left
      int guard = 0, maxsteps = 8 * W * H + 16;
      do {
        contour.push_back({(double)cx, (double)cy});
        int found = -1;
        for (int k = 1; k <= 8; ++k) {
          int d = (bdir + k) & 7;
          int nx = cx + dx8[d], ny = cy + dy8[d];
          if (fg(nx, ny)) { found = d; cx = nx; cy = ny; break; }
        }
        if (found < 0) break;                                // isolated pixel
        bdir = (found + 5) & 7;                              // step back one, then scan forward
        if (++guard > maxsteps) break;
      } while (!(cx == x && cy == y));
      if (contour.size() >= 3) out.push_back(contour);
      else if (!contour.empty()) out.push_back(contour);

      // flood fill the whole component so it is not traced again
      stack.clear();
      stack.push_back((int)idx);
      seen[idx] = 1;
      while (!stack.empty()) {
        int p = stack.back(); stack.pop_back();
        int px = p % W, py = p / W;
        for (int d = 0; d < 8; ++d) {
          int nx = px + dx8[d], ny = py + dy8[d];
          if (!fg(nx, ny)) continue;
          size_t ni = (size_t)ny * W + nx;
          if (seen[ni]) continue;
          seen[ni] = 1;
          stack.push_back((int)ni);
        }
      }
      if ((int)out.size() >= 100000) return out;             // pathological input guard
    }
  return out;
}

// ---- the whole post-process -------------------------------------------------------------------
// `prob` is the graph's [1,1,H,W] output; the boxes come back in the ORIGINAL image's pixels.
inline std::vector<Box> boxes_from_prob(const Tensor& prob, int src_w, int src_h, const Cfg& cfg) {
  int H = (int)prob->shape[2], W = (int)prob->shape[3];
  const float* p = prob->data.data();
  std::vector<unsigned char> bin((size_t)W * H, 0);
  for (size_t i = 0; i < bin.size(); ++i) bin[i] = p[i] > cfg.thresh ? 1 : 0;

  std::vector<Box> boxes;
  auto contours = trace_contours(bin, W, H);
  int used = 0;
  for (auto& c : contours) {
    if (used >= cfg.max_candidates) break;
    ++used;
    if (c.size() < 3) continue;
    double eps = 0.002 * poly_len(c);
    std::vector<Pt> poly = approx_poly(c, eps);
    if (poly.size() < 4) continue;

    MinRect r0 = min_area_rect(poly);
    if (std::min(r0.a, r0.b) * 2 < cfg.min_size) continue;

    float score = poly_score(p, W, H, poly);
    if (score < cfg.box_thresh) continue;

    double area = poly_area(poly), len = poly_len(poly);
    if (len < 1e-6) continue;
    double dist = area * cfg.unclip_ratio / len;

    Pt q[4];
    rect_corners(min_area_rect(poly, dist), dist, q);
    double w1 = std::hypot(q[1].x - q[0].x, q[1].y - q[0].y);
    double h1 = std::hypot(q[3].x - q[0].x, q[3].y - q[0].y);
    if (std::min(w1, h1) < cfg.min_size + 2) continue;

    Box b;
    b.score = score;
    for (int i = 0; i < 4; ++i) {
      b.q[i].x = std::min(std::max(std::round(q[i].x / W * src_w), 0.0), (double)src_w);
      b.q[i].y = std::min(std::max(std::round(q[i].y / H * src_h), 0.0), (double)src_h);
    }
    boxes.push_back(b);
  }
  return boxes;
}

// sorted_boxes: reading order — top to bottom, then left to right within a 10 px band so a wobbly
// line does not get shuffled.
//
// Two passes, exactly as PaddleOCR does it, and NOT a single std::sort with a "within 10 px, compare
// x" comparator: that comparator is not transitive (a~b, b~c, but a<c strictly), which is undefined
// behaviour for std::sort and in practice reorders boxes differently from the reference.
inline void sort_boxes(std::vector<Box>& b) {
  std::sort(b.begin(), b.end(), [](const Box& l, const Box& r) {
    if (l.q[0].y != r.q[0].y) return l.q[0].y < r.q[0].y;
    return l.q[0].x < r.q[0].x;
  });
  for (size_t i = 0; i + 1 < b.size(); ++i)
    for (size_t j = i + 1; j-- > 0;) {
      if (std::fabs(b[j + 1].q[0].y - b[j].q[0].y) < 10 && b[j + 1].q[0].x < b[j].q[0].x)
        std::swap(b[j], b[j + 1]);
      else
        break;
    }
}

}  // namespace db
