// Single-header device backend seam. Same kernel source runs on CUDA (nvcc,
// -DUSE_CUDA) or on the CPU (g++/MSVC, no GPU needed — the CPU path IS the
// "emulation"). Kernels are written once as `bk::parallel_for(n, [=] BK_HD (i){...})`.
//
//   CPU  : g++  -std=c++20 -O2 [-fopenmp]              pure/m17_gpu.cpp
//   CUDA : nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -O2 pure/m17_gpu.cpp
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#ifdef USE_CUDA
  #ifdef USE_CUBLAS
    #include <cublas_v2.h>        // include FIRST (before anything pulls cuda_fp16 in a bad state)
  #endif
  #include <cuda_runtime.h>
  #define BK_HD __host__ __device__
  #define BK_CHECK(x) do { cudaError_t e_=(x); if (e_) { \
      fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); std::abort(); } } while(0)
#else
  #include "parallel.hpp"          // CPU parallel_for (OpenMP / std::thread)
  #define BK_HD
  #ifdef USE_EIGEN
    #include "Eigen_Core.h"        // vendored flat Eigen (GEMM). build: -Ipure/third_party/eigen_flat -DUSE_EIGEN (and /arch:AVX2 or -march=native)
  #endif
#endif

namespace bk {

inline const char* backend_name() {
#ifdef USE_CUDA
  return "CUDA (GPU)";
#else
  return "CPU (emulation — no GPU)";
#endif
}

// Owning device buffer. On CPU it is just host memory; on CUDA it is device memory
// with explicit host<->device copies, so call sites are already GPU-shaped.
template <class T>
struct Buffer {
  T* p_ = nullptr; int64_t n_ = 0;
  Buffer() = default;
  explicit Buffer(int64_t n) { alloc(n); }
  Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { free_(); }

  T* data() { return p_; }
  const T* data() const { return p_; }
  int64_t size() const { return n_; }

#ifdef USE_CUDA
  void alloc(int64_t n) { free_(); n_ = n; BK_CHECK(cudaMalloc(&p_, n * sizeof(T))); }
  void from_host(const T* h, int64_t n) { alloc(n); BK_CHECK(cudaMemcpy(p_, h, n*sizeof(T), cudaMemcpyHostToDevice)); }
  void to_host(T* h) const { BK_CHECK(cudaMemcpy(h, p_, n_*sizeof(T), cudaMemcpyDeviceToHost)); }
  void free_() { if (p_) cudaFree(p_); p_ = nullptr; }
#else
  void alloc(int64_t n) { free_(); n_ = n; p_ = (T*)std::malloc(n * sizeof(T)); }
  void from_host(const T* h, int64_t n) { alloc(n); std::memcpy(p_, h, n * sizeof(T)); }
  void to_host(T* h) const { std::memcpy(h, p_, n_ * sizeof(T)); }
  void free_() { std::free(p_); p_ = nullptr; }
#endif
};

#ifdef USE_CUDA
template <class F> __global__ void _grid_kernel(int64_t n, F f) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}
template <class F> inline void parallel_for(int64_t n, F f) {
  if (n <= 0) return;
  const int block = 256;
  int64_t grid = (n + block - 1) / block;
  _grid_kernel<<<(unsigned)grid, block>>>(n, f);
  BK_CHECK(cudaGetLastError());
  BK_CHECK(cudaDeviceSynchronize());
}
inline void sync() { BK_CHECK(cudaDeviceSynchronize()); }
#else
template <class F> inline void parallel_for(int64_t n, F f) {
  ::parallel_for(n, f);            // reuse the CPU parallel_for
}
inline void sync() {}
#endif

// GEMM family, all computing  C = op(A) * op(B) + beta*C  (beta=0 overwrites).
// Pointers live in the active memory space (device under CUDA, host on CPU).
//   gemm    : C(M,N) = A(M,K)   * B(K,N)                 (NN)
//   gemm_nt : C(M,N) = A(M,Kc)  * B(N,Kc)^T              (NT) — conv dW = dO * col^T
//   gemm_tn : C(M,N) = A(Kc,M)^T* B(Kc,N)               (TN) — conv dcol = W^T * dO
#ifdef USE_CUDA
// Optional cuBLAS fast path (build with -DUSE_CUBLAS -lcublas). cuBLAS is column-major; our
// buffers are row-major, so each row-major GEMM maps to a cuBLAS call computing Cᵀ = ... :
//   C=A·B      -> sgemm(N,N, N,M,K,  B(ld N), A(ld K), C(ld N))
//   C=A·Bᵀ(nt) -> sgemm(T,N, N,M,Kc, B(ld Kc), A(ld Kc), C(ld N))
//   C=Aᵀ·B(tn) -> sgemm(N,T, N,M,Kc, B(ld N),  A(ld M),  C(ld N))
#ifdef USE_CUBLAS
inline cublasHandle_t bk_cublas() { static cublasHandle_t h = nullptr; if (!h) cublasCreate(&h); return h; }
#endif
inline void gemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N, float beta = 0.f) {
#ifdef USE_CUBLAS
  float alpha = 1.f;
  cublasSgemm(bk_cublas(), CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, B, (int)N, A, (int)K, &beta, C, (int)N);
#else
  bk::parallel_for(M * N, [=] BK_HD (int64_t idx) {
    int64_t m = idx / N, n = idx % N; float s = 0.f;
    for (int64_t k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
    C[idx] = s + (beta == 0.f ? 0.f : beta * C[idx]);
  });
#endif
}
inline void gemm_nt(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
#ifdef USE_CUBLAS
  float alpha = 1.f;
  cublasSgemm(bk_cublas(), CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)Kc, &alpha, B, (int)Kc, A, (int)Kc, &beta, C, (int)N);
#else
  bk::parallel_for(M * N, [=] BK_HD (int64_t idx) {
    int64_t m = idx / N, n = idx % N; const float* ar = A + m * Kc; const float* br = B + n * Kc;
    float s = 0.f; for (int64_t k = 0; k < Kc; ++k) s += ar[k] * br[k];
    C[idx] = s + (beta == 0.f ? 0.f : beta * C[idx]);
  });
#endif
}
inline void gemm_tn(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
#ifdef USE_CUBLAS
  float alpha = 1.f;
  cublasSgemm(bk_cublas(), CUBLAS_OP_N, CUBLAS_OP_T, (int)N, (int)M, (int)Kc, &alpha, B, (int)N, A, (int)M, &beta, C, (int)N);
#else
  bk::parallel_for(M * N, [=] BK_HD (int64_t idx) {
    int64_t m = idx / N, n = idx % N; float s = 0.f;
    for (int64_t k = 0; k < Kc; ++k) s += A[k * M + m] * B[k * N + n];
    C[idx] = s + (beta == 0.f ? 0.f : beta * C[idx]);
  });
#endif
}
#elif defined(USE_EIGEN)
// Eigen GEMM backend (CPU, opt-in): the same im2col+GEMM conv, but the GEMM is Eigen's
// blocked + SIMD kernel — ~10x+ over the plain loops below. Row-major float maps; beta
// gives C = A·B + beta·C. Build: -DUSE_EIGEN -Ipure/third_party/eigen_flat (+ AVX flag).
namespace eig_ { using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                 using Map = Eigen::Map<RM>; using CMap = Eigen::Map<const RM>; }
inline void gemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N, float beta = 0.f) {
  eig_::CMap Am(A, M, K), Bm(B, K, N); eig_::Map Cm(C, M, N);
  if (beta == 0.f) Cm.noalias() = Am * Bm;
  else { if (beta != 1.f) Cm *= beta; Cm.noalias() += Am * Bm; }
}
inline void gemm_nt(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  eig_::CMap Am(A, M, Kc), Bm(B, N, Kc); eig_::Map Cm(C, M, N);
  if (beta == 0.f) Cm.noalias() = Am * Bm.transpose();
  else { if (beta != 1.f) Cm *= beta; Cm.noalias() += Am * Bm.transpose(); }
}
inline void gemm_tn(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  eig_::CMap Am(A, Kc, M), Bm(B, Kc, N); eig_::Map Cm(C, M, N);
  if (beta == 0.f) Cm.noalias() = Am.transpose() * Bm;
  else { if (beta != 1.f) Cm *= beta; Cm.noalias() += Am.transpose() * Bm; }
}
#else
inline void gemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N, float beta = 0.f) {
  bk::parallel_for(M, [=] (int64_t m) {
    float* cr = C + m * N;
    if (beta == 0.f) for (int64_t n = 0; n < N; ++n) cr[n] = 0.f;
    else if (beta != 1.f) for (int64_t n = 0; n < N; ++n) cr[n] *= beta;
    for (int64_t k = 0; k < K; ++k) { float av = A[m * K + k]; const float* br = B + k * N;
      for (int64_t n = 0; n < N; ++n) cr[n] += av * br[n]; }
  });
}
inline void gemm_nt(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  bk::parallel_for(M, [=] (int64_t m) {
    const float* ar = A + m * Kc; float* cr = C + m * N;
    for (int64_t n = 0; n < N; ++n) {
      const float* br = B + n * Kc; float s = 0.f;
      for (int64_t k = 0; k < Kc; ++k) s += ar[k] * br[k];
      cr[n] = s + (beta == 0.f ? 0.f : beta * cr[n]);
    }
  });
}
inline void gemm_tn(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  bk::parallel_for(M, [=] (int64_t m) {
    float* cr = C + m * N;
    if (beta == 0.f) for (int64_t n = 0; n < N; ++n) cr[n] = 0.f;
    else if (beta != 1.f) for (int64_t n = 0; n < N; ++n) cr[n] *= beta;
    for (int64_t k = 0; k < Kc; ++k) { float av = A[k * M + m]; const float* br = B + k * N;
      for (int64_t n = 0; n < N; ++n) cr[n] += av * br[n]; }
  });
}
#endif

// Host-facing wrappers: host pointers in/out. On CUDA they stage to the device
// (staging C in too when beta!=0), run, and copy back; on CPU they run in place.
// This is the single place the engine's ops route through to reach the device.
#ifdef USE_CUDA
  #define BK_STAGE_C(C, n) if (beta != 0.f) dC.from_host((C), (n)); else dC.alloc((n))
inline void gemm_hosted(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N, float beta = 0.f) {
  Buffer<float> dA, dB, dC; dA.from_host(A, M*K); dB.from_host(B, K*N); BK_STAGE_C(C, M*N);
  gemm(dA.data(), dB.data(), dC.data(), M, K, N, beta); dC.to_host(C);
}
inline void gemm_nt_hosted(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  Buffer<float> dA, dB, dC; dA.from_host(A, M*Kc); dB.from_host(B, N*Kc); BK_STAGE_C(C, M*N);
  gemm_nt(dA.data(), dB.data(), dC.data(), M, N, Kc, beta); dC.to_host(C);
}
inline void gemm_tn_hosted(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) {
  Buffer<float> dA, dB, dC; dA.from_host(A, Kc*M); dB.from_host(B, Kc*N); BK_STAGE_C(C, M*N);
  gemm_tn(dA.data(), dB.data(), dC.data(), M, N, Kc, beta); dC.to_host(C);
}
  #undef BK_STAGE_C
#else
inline void gemm_hosted(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N, float beta = 0.f) { gemm(A, B, C, M, K, N, beta); }
inline void gemm_nt_hosted(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) { gemm_nt(A, B, C, M, N, Kc, beta); }
inline void gemm_tn_hosted(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t Kc, float beta = 0.f) { gemm_tn(A, B, C, M, N, Kc, beta); }
#endif

} // namespace bk
