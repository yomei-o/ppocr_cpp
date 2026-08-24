// Extra differentiable ops for the loss: elementwise binary/unary, row-softmax,
// column ops. Tensors are treated as (rows, cols) with cols = shape.back().
#pragma once
#include "autograd.hpp"

inline int64_t last_dim(const Tensor& a) { return a->shape.back(); }

// ---- elementwise same-shape binary ----
#define EW_BIN(NAME, FWD, GA, GB)                                              \
inline Tensor NAME(const Tensor& a, const Tensor& b) {                         \
  auto o = make_tensor(a->shape, true);                                        \
  for (int64_t i = 0; i < o->numel(); ++i) { float x=a->data[i], y=b->data[i]; o->data[i] = (FWD); } \
  o->parents = {a, b}; Node* op = o.get();                                     \
  o->backward_fn = [a, b, op] { for (int64_t i=0;i<op->numel();++i){           \
      float x=a->data[i], y=b->data[i], g=op->grad[i]; a->grad[i]+=(GA); b->grad[i]+=(GB); } }; \
  return o; }
EW_BIN(sub,  x - y,           g,            -g)
EW_BIN(divi, x / y,           g / y,        -g * x / (y * y))
EW_BIN(max2, x > y ? x : y,   (x >= y ? g : 0.f), (x < y ? g : 0.f))
EW_BIN(min2, x < y ? x : y,   (x <= y ? g : 0.f), (x > y ? g : 0.f))
#undef EW_BIN

// ---- unary ----
#define EW_UN(NAME, FWD, DERIV)                                                \
inline Tensor NAME(const Tensor& a) {                                          \
  auto o = make_tensor(a->shape, true);                                        \
  for (int64_t i=0;i<o->numel();++i){ float x=a->data[i]; o->data[i]=(FWD); }  \
  o->parents={a}; Node* op=o.get();                                            \
  o->backward_fn=[a,op]{ for(int64_t i=0;i<op->numel();++i){ float x=a->data[i]; a->grad[i]+=(DERIV)*op->grad[i]; } }; \
  return o; }
EW_UN(op_exp,   std::exp(x),        std::exp(x))
EW_UN(op_log,   std::log(x),        1.f / x)
EW_UN(op_atan,  std::atan(x),       1.f / (1.f + x * x))
EW_UN(op_abs,   std::abs(x),        (x > 0 ? 1.f : (x < 0 ? -1.f : 0.f)))
EW_UN(op_log1p, std::log1p(x),      1.f / (1.f + x))
#undef EW_UN

inline Tensor add_scalar(const Tensor& a, float c) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i=0;i<o->numel();++i) o->data[i]=a->data[i]+c;
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op]{ for(int64_t i=0;i<op->numel();++i) a->grad[i]+=op->grad[i]; };
  return o;
}
inline Tensor mul_scalar(const Tensor& a, float c) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i=0;i<o->numel();++i) o->data[i]=a->data[i]*c;
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,c]{ for(int64_t i=0;i<op->numel();++i) a->grad[i]+=c*op->grad[i]; };
  return o;
}
inline Tensor clampmin_scalar(const Tensor& a, float c) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i=0;i<o->numel();++i) o->data[i]=a->data[i]<c?c:a->data[i];
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,c]{ for(int64_t i=0;i<op->numel();++i) a->grad[i]+=(a->data[i]>c?op->grad[i]:0.f); };
  return o;
}
inline Tensor detach(const Tensor& a) {   // constant snapshot (stops gradient)
  return from_data(a->shape, a->data, false);
}

// ---- reshape (same data) ----
inline Tensor reshape(const Tensor& a, std::vector<int64_t> shape) {
  auto o = make_tensor(shape, true); o->data = a->data;
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op]{ for(int64_t i=0;i<op->numel();++i) a->grad[i]+=op->grad[i]; };
  return o;
}

// ---- column ops (2D view: rows x cols) ----
inline Tensor narrow_col(const Tensor& a, int64_t j) {   // -> (rows,1)
  int64_t C=last_dim(a), R=a->numel()/C;
  auto sh=a->shape; sh.back()=1; auto o=make_tensor(sh,true);
  for(int64_t r=0;r<R;++r) o->data[r]=a->data[r*C+j];
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,R,C,j]{ for(int64_t r=0;r<R;++r) a->grad[r*C+j]+=op->grad[r]; };
  return o;
}
inline Tensor concat_cols(const std::vector<Tensor>& xs) { // each (rows,1) -> (rows,k)
  int64_t R=xs[0]->numel(), K=(int64_t)xs.size();
  auto sh=xs[0]->shape; sh.back()=K; auto o=make_tensor(sh,true);
  for(int64_t r=0;r<R;++r) for(int64_t k=0;k<K;++k) o->data[r*K+k]=xs[k]->data[r];
  o->parents=xs; Node* op=o.get();
  o->backward_fn=[xs,op,R,K]{ for(int64_t r=0;r<R;++r) for(int64_t k=0;k<K;++k) xs[k]->grad[r]+=op->grad[r*K+k]; };
  return o;
}

// ---- row softmax + weighted reduce over cols ----
inline Tensor softmax_rows(const Tensor& a) {
  int64_t C=last_dim(a), R=a->numel()/C;
  auto o=make_tensor(a->shape,true);
  for(int64_t r=0;r<R;++r){
    float m=-1e30f; for(int64_t c=0;c<C;++c) m=std::max(m,a->data[r*C+c]);
    double s=0; for(int64_t c=0;c<C;++c){ float e=std::exp(a->data[r*C+c]-m); o->data[r*C+c]=e; s+=e; }
    for(int64_t c=0;c<C;++c) o->data[r*C+c]/=(float)s;
  }
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,R,C]{ for(int64_t r=0;r<R;++r){
      double dot=0; for(int64_t c=0;c<C;++c) dot+=(double)op->grad[r*C+c]*op->data[r*C+c];
      for(int64_t c=0;c<C;++c){ float y=op->data[r*C+c]; a->grad[r*C+c]+=y*(op->grad[r*C+c]-(float)dot); } } };
  return o;
}
inline Tensor log_softmax_rows(const Tensor& a) {
  int64_t C=last_dim(a), R=a->numel()/C;
  auto o=make_tensor(a->shape,true);
  for(int64_t r=0;r<R;++r){
    float m=-1e30f; for(int64_t c=0;c<C;++c) m=std::max(m,a->data[r*C+c]);
    double s=0; for(int64_t c=0;c<C;++c) s+=std::exp(a->data[r*C+c]-m);
    float lse=m+(float)std::log(s);
    for(int64_t c=0;c<C;++c) o->data[r*C+c]=a->data[r*C+c]-lse;
  }
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,R,C]{ for(int64_t r=0;r<R;++r){
      double gsum=0; for(int64_t c=0;c<C;++c) gsum+=op->grad[r*C+c];
      for(int64_t c=0;c<C;++c){ float y=std::exp(op->data[r*C+c]); a->grad[r*C+c]+=op->grad[r*C+c]-y*(float)gsum; } } };
  return o;
}
// gather one column per row by (constant) index -> (rows,1)
inline Tensor gather_row(const Tensor& a, const std::vector<int64_t>& idx) {
  int64_t C=last_dim(a), R=a->numel()/C;
  auto sh=a->shape; sh.back()=1; auto o=make_tensor(sh,true);
  for(int64_t r=0;r<R;++r) o->data[r]=a->data[r*C+idx[r]];
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,R,C,idx]{ for(int64_t r=0;r<R;++r) a->grad[r*C+idx[r]]+=op->grad[r]; };
  return o;
}
inline Tensor wsum_rows(const Tensor& a, const std::vector<float>& coeff) { // -> (rows,1)
  int64_t C=last_dim(a), R=a->numel()/C;
  auto sh=a->shape; sh.back()=1; auto o=make_tensor(sh,true);
  for(int64_t r=0;r<R;++r){ double s=0; for(int64_t c=0;c<C;++c) s+=(double)a->data[r*C+c]*coeff[c]; o->data[r]=(float)s; }
  o->parents={a}; Node* op=o.get();
  o->backward_fn=[a,op,R,C,coeff]{ for(int64_t r=0;r<R;++r) for(int64_t c=0;c<C;++c) a->grad[r*C+c]+=op->grad[r]*coeff[c]; };
  return o;
}
