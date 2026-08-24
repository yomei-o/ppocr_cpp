// Image container + the exact preprocessing PaddleOCR feeds these two graphs, plus the rotated-quad
// crop between them. Nothing here is generic image processing for its own sake: every constant is
// transcribed from PaddleOCR so the C++ pipeline sees the same pixels the Python one does.
//
// CHANNEL ORDER IS BGR, NOT RGB. Both PP-OCR models were trained through
// `DecodeImage: {img_mode: BGR}` and the NormalizeImage means are applied *positionally* to that
// BGR array — so channel 0 gets mean 0.485. Feeding RGB is the classic silent-wrong: detection
// still finds most boxes and recognition still emits plausible Japanese, just worse. Do not "fix"
// the order without re-measuring.
#pragma once
#include "autograd.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace im {

struct Img {
  int w = 0, h = 0, c = 3;              // c is 3 (BGR) unless noted
  std::vector<unsigned char> d;
  unsigned char at(int y, int x, int k) const { return d[((size_t)y * w + x) * c + k]; }
  bool empty() const { return w <= 0 || h <= 0; }
};

inline Img make_img(int w, int h, int c = 3) {
  Img o; o.w = w; o.h = h; o.c = c; o.d.assign((size_t)w * h * c, 0);
  return o;
}

// RGBA (canvas order, what the browser hands over) -> BGR
inline Img from_rgba(const unsigned char* rgba, int w, int h) {
  Img o = make_img(w, h, 3);
  for (int i = 0; i < w * h; ++i) {
    o.d[(size_t)i * 3 + 0] = rgba[(size_t)i * 4 + 2];
    o.d[(size_t)i * 3 + 1] = rgba[(size_t)i * 4 + 1];
    o.d[(size_t)i * 3 + 2] = rgba[(size_t)i * 4 + 0];
  }
  return o;
}

// ---- bilinear resize (cv2.resize INTER_LINEAR, same half-pixel centres) ----
inline Img resize(const Img& s, int nw, int nh) {
  Img o = make_img(nw, nh, s.c);
  if (s.empty() || nw <= 0 || nh <= 0) return o;
  double fx = (double)s.w / nw, fy = (double)s.h / nh;
  for (int y = 0; y < nh; ++y) {
    double sy = (y + 0.5) * fy - 0.5;
    int y0 = (int)std::floor(sy);
    double wy = sy - y0;
    int y0c = std::min(std::max(y0, 0), s.h - 1), y1c = std::min(std::max(y0 + 1, 0), s.h - 1);
    for (int x = 0; x < nw; ++x) {
      double sx = (x + 0.5) * fx - 0.5;
      int x0 = (int)std::floor(sx);
      double wx = sx - x0;
      int x0c = std::min(std::max(x0, 0), s.w - 1), x1c = std::min(std::max(x0 + 1, 0), s.w - 1);
      for (int k = 0; k < s.c; ++k) {
        double v = s.at(y0c, x0c, k) * (1 - wx) * (1 - wy) + s.at(y0c, x1c, k) * wx * (1 - wy)
                 + s.at(y1c, x0c, k) * (1 - wx) * wy + s.at(y1c, x1c, k) * wx * wy;
        o.d[((size_t)y * nw + x) * o.c + k] = (unsigned char)std::lround(std::min(255.0, std::max(0.0, v)));
      }
    }
  }
  return o;
}

// ---- detector input ---------------------------------------------------------------------------
// DetResizeForTest(limit_type='max', limit_side_len=960): shrink so the long side fits, never
// enlarge, then round both sides to a multiple of 32. Keep the two ratios — the boxes come back in
// resized coordinates and have to be mapped home.
struct DetPre {
  Tensor x;                 // [1,3,H,W]
  int rw = 0, rh = 0;       // resized size actually fed
  double ratio_w = 1, ratio_h = 1;   // resized / original
};

inline DetPre det_input(const Img& src, int limit_side_len = 960, bool limit_max = true) {
  int h = src.h, w = src.w;
  double ratio = 1.0;
  if (limit_max) {
    if (std::max(h, w) > limit_side_len) ratio = (double)limit_side_len / std::max(h, w);
  } else {
    if (std::min(h, w) < limit_side_len) ratio = (double)limit_side_len / std::min(h, w);
  }
  int rh = (int)std::lround(h * ratio), rw = (int)std::lround(w * ratio);
  rh = std::max(32, (int)std::round(rh / 32.0) * 32);
  rw = std::max(32, (int)std::round(rw / 32.0) * 32);
  Img r = resize(src, rw, rh);

  static const float MEAN[3] = {0.485f, 0.456f, 0.406f};    // positional over BGR, see header note
  static const float STD[3] = {0.229f, 0.224f, 0.225f};
  DetPre out;
  out.x = make_tensor({1, 3, rh, rw}, false);
  out.x->grad.clear();
  for (int k = 0; k < 3; ++k)
    for (int y = 0; y < rh; ++y)
      for (int x = 0; x < rw; ++x)
        out.x->data[((size_t)k * rh + y) * rw + x] = (r.at(y, x, k) / 255.f - MEAN[k]) / STD[k];
  out.rw = rw; out.rh = rh;
  out.ratio_w = (double)rw / w;
  out.ratio_h = (double)rh / h;
  return out;
}

// ---- recognizer input -------------------------------------------------------------------------
// resize_norm_img: height 48, width = ceil(48 * w/h) (the batch form, no 320 cap), then
// (x/255 - 0.5) / 0.5. Width is padded on the right with zeros up to `pad_w` when batching.
inline Tensor rec_input(const Img& crop, int img_h = 48, int pad_w = 0) {
  int w = std::max(1, (int)std::ceil((double)img_h * crop.w / std::max(1, crop.h)));
  int W = pad_w > 0 ? pad_w : w;
  if (w > W) w = W;
  Img r = resize(crop, w, img_h);
  Tensor t = make_tensor({1, 3, img_h, W}, false);
  t->grad.clear();
  for (int k = 0; k < 3; ++k)
    for (int y = 0; y < img_h; ++y)
      for (int x = 0; x < w; ++x)
        t->data[((size_t)k * img_h + y) * W + x] = (r.at(y, x, k) / 255.f - 0.5f) / 0.5f;
  return t;
}

// ---- perspective crop of a rotated quad -------------------------------------------------------
struct Pt { double x = 0, y = 0; };

// Solve the 8x8 system for the homography mapping dst -> src (we sample backwards).
inline bool persp_from_quad(const Pt s[4], const Pt d[4], double H[9]) {
  double A[8][9] = {{0}};
  for (int i = 0; i < 4; ++i) {
    double* r0 = A[i * 2];
    r0[0] = d[i].x; r0[1] = d[i].y; r0[2] = 1;
    r0[3] = r0[4] = r0[5] = 0;
    r0[6] = -d[i].x * s[i].x; r0[7] = -d[i].y * s[i].x; r0[8] = s[i].x;
    double* r1 = A[i * 2 + 1];
    r1[0] = r1[1] = r1[2] = 0;
    r1[3] = d[i].x; r1[4] = d[i].y; r1[5] = 1;
    r1[6] = -d[i].x * s[i].y; r1[7] = -d[i].y * s[i].y; r1[8] = s[i].y;
  }
  for (int col = 0; col < 8; ++col) {                       // Gauss-Jordan with partial pivoting
    int piv = col;
    for (int r = col + 1; r < 8; ++r) if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
    if (std::fabs(A[piv][col]) < 1e-12) return false;
    if (piv != col) for (int k = 0; k <= 8; ++k) std::swap(A[col][k], A[piv][k]);
    double p = A[col][col];
    for (int k = col; k <= 8; ++k) A[col][k] /= p;
    for (int r = 0; r < 8; ++r) {
      if (r == col) continue;
      double f = A[r][col];
      if (f == 0) continue;
      for (int k = col; k <= 8; ++k) A[r][k] -= f * A[col][k];
    }
  }
  for (int i = 0; i < 8; ++i) H[i] = A[i][8];
  H[8] = 1.0;
  return true;
}

// get_rotate_crop_image: the quad's own side lengths decide the output size, and a crop taller than
// 1.5x its width is rotated 90 deg CCW (vertical Japanese text arrives that way).
inline Img crop_quad(const Img& src, const Pt q[4], bool rotate_tall = true) {
  auto dist = [](const Pt& a, const Pt& b) { return std::hypot(a.x - b.x, a.y - b.y); };
  int cw = (int)std::lround(std::max(dist(q[0], q[1]), dist(q[2], q[3])));
  int ch = (int)std::lround(std::max(dist(q[0], q[3]), dist(q[1], q[2])));
  cw = std::max(1, cw); ch = std::max(1, ch);
  Pt d[4] = {{0, 0}, {(double)cw, 0}, {(double)cw, (double)ch}, {0, (double)ch}};
  double H[9];
  Img out = make_img(cw, ch, src.c);
  if (!persp_from_quad(q, d, H)) return out;
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      double den = H[6] * x + H[7] * y + H[8];
      if (std::fabs(den) < 1e-12) continue;
      double sx = (H[0] * x + H[1] * y + H[2]) / den;
      double sy = (H[3] * x + H[4] * y + H[5]) / den;
      int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
      double wx = sx - x0, wy = sy - y0;
      int x0c = std::min(std::max(x0, 0), src.w - 1), x1c = std::min(std::max(x0 + 1, 0), src.w - 1);
      int y0c = std::min(std::max(y0, 0), src.h - 1), y1c = std::min(std::max(y0 + 1, 0), src.h - 1);
      for (int k = 0; k < src.c; ++k) {
        double v = src.at(y0c, x0c, k) * (1 - wx) * (1 - wy) + src.at(y0c, x1c, k) * wx * (1 - wy)
                 + src.at(y1c, x0c, k) * (1 - wx) * wy + src.at(y1c, x1c, k) * wx * wy;
        out.d[((size_t)y * cw + x) * out.c + k] =
            (unsigned char)std::lround(std::min(255.0, std::max(0.0, v)));
      }
    }
  if (rotate_tall && (double)ch / cw >= 1.5) {              // rotate 90 CCW
    Img rot = make_img(ch, cw, out.c);
    for (int y = 0; y < cw; ++y)
      for (int x = 0; x < ch; ++x)
        for (int k = 0; k < out.c; ++k)
          rot.d[((size_t)y * ch + x) * out.c + k] = out.at(x, cw - 1 - y, k);
    return rot;
  }
  return out;
}

// ---- drawing (for the CLI's annotated output) -------------------------------------------------
inline void draw_line(Img& img, double x0, double y0, double x1, double y1,
                      unsigned char b, unsigned char g, unsigned char r, int thick = 2) {
  int n = (int)std::lround(std::max(std::fabs(x1 - x0), std::fabs(y1 - y0))) + 1;
  for (int i = 0; i <= n; ++i) {
    double t = (double)i / n;
    int px = (int)std::lround(x0 + (x1 - x0) * t), py = (int)std::lround(y0 + (y1 - y0) * t);
    for (int dy = -thick / 2; dy <= thick / 2; ++dy)
      for (int dx = -thick / 2; dx <= thick / 2; ++dx) {
        int x = px + dx, y = py + dy;
        if (x < 0 || y < 0 || x >= img.w || y >= img.h) continue;
        size_t o = ((size_t)y * img.w + x) * img.c;
        img.d[o + 0] = b; img.d[o + 1] = g; img.d[o + 2] = r;
      }
  }
}

inline void draw_quad(Img& img, const Pt q[4], unsigned char b, unsigned char g, unsigned char r,
                      int thick = 2) {
  for (int i = 0; i < 4; ++i)
    draw_line(img, q[i].x, q[i].y, q[(i + 1) % 4].x, q[(i + 1) % 4].y, b, g, r, thick);
}

}  // namespace im
