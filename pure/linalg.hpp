// Dense linear-algebra ops for the pure engine: 2D matmul and transpose (with
// backward). These are what yolov11's C2PSA attention needs on top of the existing
// softmax. Tensors are treated as 2D (rows = product of leading dims, cols = last dim).
#pragma once
#include "autograd.hpp"

// A (M,K) x B (K,N) -> (M,N)
inline Tensor matmul(const Tensor& A, const Tensor& B) {
  int64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
  assert(B->shape[0] == K);
  auto o = make_tensor({M, N}, true);
  bk::gemm_hosted(A->data.data(), B->data.data(), o->data.data(), M, K, N);  // device seam
  o->parents = {A, B};
  Node* op = o.get();
  o->backward_fn = [A, B, op, M, K, N] {
    const float* a = A->data.data(); const float* b = B->data.data(); const float* go = op->grad.data();
    float* ga = A->grad.data(); float* gb = B->grad.data();
    for (int64_t m = 0; m < M; ++m)
      for (int64_t k = 0; k < K; ++k) {
        float s = 0; const float* brow = &b[k * N]; const float* grow = &go[m * N];
        for (int64_t n = 0; n < N; ++n) s += grow[n] * brow[n];
        ga[m * K + k] += s;                                   // dA = dO @ B^T
      }
    for (int64_t k = 0; k < K; ++k)
      for (int64_t m = 0; m < M; ++m) {
        float av = a[m * K + k]; const float* grow = &go[m * N]; float* gbrow = &gb[k * N];
        for (int64_t n = 0; n < N; ++n) gbrow[n] += av * grow[n];   // dB = A^T @ dO
      }
  };
  return o;
}

// transpose of a 2D tensor (M,N) -> (N,M)
inline Tensor transpose2d(const Tensor& A) {
  int64_t M = A->shape[0], N = A->shape[1];
  auto o = make_tensor({N, M}, true);
  for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) o->data[n * M + m] = A->data[m * N + n];
  o->parents = {A};
  Node* op = o.get();
  o->backward_fn = [A, op, M, N] {
    for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) A->grad[m * N + n] += op->grad[n * M + m];
  };
  return o;
}
