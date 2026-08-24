eigen_flat — minimal Eigen (GEMM only), FLAT: files only, no subfolders
=======================================================================

130 Eigen 3.4.0 headers (the closure that <Eigen/Core> pulls for dense float matmul,
incl. transpose products + the array/row ops a conv backend needs) + 2 license files.
Every file lives at the top level; each is renamed to its original relative path with
'/' -> '_' (so AVX vs SSE PacketMath.h don't collide), and all #includes were rewritten
to match. Copy this ONE folder and go.

USE
---
  #include "Eigen_Core.h"          // <- the entry header (was <Eigen/Core>)
  using RowMat = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  Eigen::Map<const RowMat> W(w,Cout,K), C(col,K,HW);
  Eigen::Map<RowMat>       Y(y,Cout,HW);
  Y.noalias() = W * C;             // conv2d = im2col -> this GEMM

BUILD  (MUST enable SIMD or you lose ~10x)
  MSVC : cl  /std:c++17 /O2 /EHsc /arch:AVX2 /Ieigen_flat  your.cpp
  g++  : g++ -O3 -std=c++17 -march=native -Ieigen_flat      your.cpp

NOTES
  - x86 only (AVX/SSE/Default arch backends). ARM/NEON not included.
  - single-threaded by default; for multicore build with OpenMP + Eigen::setNbThreads(n).
  - if another compiler asks for a missing Eigen header, add that one file from full
    Eigen 3.4.0 (flatten its path the same way: relpath with '/'->'_').
  - License: Eigen is MPL2 (see COPYING.*). This is a subset, not a fork.
  - Verified: builds against this folder alone; conv matches a naive reference to 1.3e-6,
    ~28x faster than the naive loop (single thread, AVX2).
