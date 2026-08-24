// A fast expf, because the obvious one is not.
//
// On the mingw toolchain this repo builds with (w64devkit / gcc 14), `std::exp` on a float costs
// ~150 ns per call — roughly 100x a normal libm. Measured on this repo's sample that made the
// detector's single output Sigmoid over a 512x960 map (165 ms) and the recognizer's CTC-head
// Softmax over 39x18385 (150 ms) the two most expensive nodes in either graph, each of them more
// expensive than any convolution. Nothing about the models is at fault and no amount of GEMM
// tuning touches it.
//
// So: exp(x) = 2^k * exp(r) with k = round(x/ln2) and |r| <= ln2/2, a degree-6 Taylor series for
// exp(r), and 2^k assembled straight into the exponent field. Worst-case relative error over
// [-88, 88] measures at ~1e-7, i.e. float rounding — the parity check against onnxruntime
// (tools/parity.py) is unchanged by it at 1e-5 relative.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fm {

inline float exp_(float x) {
  if (x > 88.02f) return 3.4028235e38f;                 // overflow to ~FLT_MAX
  if (x < -103.98f) return 0.f;                         // true underflow, even for denormals
  const float INV_LN2 = 1.44269504088896341f;
  const float LN2_HI = 0.693359375f;                    // split so k*LN2 is exact in float
  const float LN2_LO = -2.12194440e-4f;
  float t = x * INV_LN2;
  float k = (t >= 0.f) ? std::floor(t + 0.5f) : std::ceil(t - 0.5f);
  float r = x - k * LN2_HI - k * LN2_LO;
  // exp(r) for |r| <= ln2/2 ~= 0.3466: the degree-6 truncation error is r^7/5040 ~= 1.2e-7
  float p = 1.f + r * (1.f + r * (0.5f + r * (0.16666667f + r * (0.041666668f +
            r * (0.008333334f + r * 0.0013888889f)))));
  int32_t ki = (int32_t)k;
  float extra = 1.f;
  if (ki < -126) {                                      // 2^k is denormal: scale in two steps
    ki += 64;
    extra = 5.4210109e-20f;                             // 2^-64
  }
  int32_t bits = (ki + 127) << 23;                      // 2^k as a float, exponent field only
  float scale;
  std::memcpy(&scale, &bits, 4);
  return p * scale * extra;
}

inline float sigmoid_(float x) { return 1.f / (1.f + exp_(-x)); }

}  // namespace fm
