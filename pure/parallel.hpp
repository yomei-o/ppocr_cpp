// Portable parallel-for. Uses OpenMP when compiled with it (g++ -fopenmp), and a
// std::thread fan-out otherwise. This keeps a single source that parallelises under
// g++/OpenMP *and* MSVC (whose OpenMP cannot parse pragmas inside the lambdas the
// autograd tape is built from) — CPU vs parallel stays a compiler-flag choice.
#pragma once
#include <cstdint>
#ifdef _OPENMP
  #include <omp.h>
#else
  #include <thread>
  #include <vector>
  #include <algorithm>
#endif

// Run body(i) for i in [0, n). body must be safe to call concurrently for distinct i.
template <class F>
inline void parallel_for(int64_t n, F body) {
  if (n <= 0) return;
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  for (int64_t i = 0; i < n; ++i) body(i);        // WASM (no threads): serial
#elif defined(PURE_SERIAL)
  // -DPURE_SERIAL: run everything on one thread. Not a debug switch — for the OCR graphs it is
  // the fastest native build there is. These ops are small (a text line is 48 x a few hundred
  // pixels) and there are hundreds of them per line, so the fan-out costs more than the work:
  // without OpenMP this function spawns a thread team per call, and with OpenMP the teams are
  // cheap but Eigen's own GEMM threading then fights for the same cores. Measured on this
  // repo's sample, recognizing 50 text lines: 18.9 s threaded -> 10.5 s serial (and 5.1 s once the
  // elementwise ops and exp were fixed too).
  for (int64_t i = 0; i < n; ++i) body(i);
#elif defined(_OPENMP)
  #pragma omp parallel for
  for (long long i = 0; i < (long long)n; ++i) body((int64_t)i);
#else
  unsigned hw = std::thread::hardware_concurrency(); if (!hw) hw = 1;
  int64_t T = std::min<int64_t>((int64_t)hw, n);
  if (T <= 1) { for (int64_t i = 0; i < n; ++i) body(i); return; }
  std::vector<std::thread> ths;
  auto worker = [&](int64_t t) { for (int64_t i = t; i < n; i += T) body(i); };
  for (int64_t t = 1; t < T; ++t) ths.emplace_back(worker, t);
  worker(0);
  for (auto& th : ths) th.join();
#endif
}
