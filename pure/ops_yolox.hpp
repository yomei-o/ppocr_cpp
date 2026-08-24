// yolox 系の追加 op（dwconv2d / slice_hw）— lpr_cpp 側の autograd に無い分を yolox_cpp から移植。
// dwconv2d は conv2d(groups=C) と等価だが、専用実装の方が速いので残す。
#pragma once
#include "autograd.hpp"

inline Tensor dwconv2d(const Tensor& in, const Tensor& w, const Tensor& bias, int64_t stride, int64_t pad) {
  int64_t N=in->shape[0], C=in->shape[1], H=in->shape[2], W=in->shape[3], k=w->shape[2];
  int64_t OH=(H+2*pad-k)/stride+1, OW=(W+2*pad-k)/stride+1;
  auto o = make_tensor({N,C,OH,OW}, true);
  const float* I=in->data.data(); const float* K=w->data.data(); const float* B=bias?bias->data.data():nullptr; float* O=o->data.data();
  #pragma omp parallel for collapse(2)
  for (int64_t n=0;n<N;++n) for (int64_t c=0;c<C;++c)
    for (int64_t oh=0;oh<OH;++oh) for (int64_t ow=0;ow<OW;++ow) {
      float acc = B?B[c]:0.f;
      for (int64_t r=0;r<k;++r){ int64_t ih=oh*stride-pad+r; if(ih<0||ih>=H)continue;
        for (int64_t s=0;s<k;++s){ int64_t iw=ow*stride-pad+s; if(iw<0||iw>=W)continue;
          acc += I[((n*C+c)*H+ih)*W+iw]*K[(c*k+r)*k+s]; } }
      O[((n*C+c)*OH+oh)*OW+ow]=acc;
    }
  o->parents = bias?std::vector<Tensor>{in,w,bias}:std::vector<Tensor>{in,w};
  Node* op=o.get();
  o->backward_fn=[in,w,bias,op,N,C,H,W,k,OH,OW,stride,pad]{
    const float* I=in->data.data(); const float* K=w->data.data(); const float* GO=op->grad.data();
    float* GI=in->grad.data(); float* GK=w->grad.data();
    #pragma omp parallel for
    for (int64_t c=0;c<C;++c)
      for (int64_t n=0;n<N;++n)
        for (int64_t oh=0;oh<OH;++oh) for (int64_t ow=0;ow<OW;++ow){
          float g=GO[((n*C+c)*OH+oh)*OW+ow]; if(g==0.f)continue;
          for (int64_t r=0;r<k;++r){ int64_t ih=oh*stride-pad+r; if(ih<0||ih>=H)continue;
            for (int64_t s=0;s<k;++s){ int64_t iw=ow*stride-pad+s; if(iw<0||iw>=W)continue;
              GI[((n*C+c)*H+ih)*W+iw]+=g*K[(c*k+r)*k+s]; GK[(c*k+r)*k+s]+=g*I[((n*C+c)*H+ih)*W+iw]; } }
        }
    if (bias){ float* GB=bias->grad.data();
      #pragma omp parallel for
      for (int64_t c=0;c<C;++c){ float a=0; for(int64_t n=0;n<N;++n) for(int64_t p=0;p<OH*OW;++p) a+=GO[((n*C+c)*OH)*OW+p]; GB[c]+=a; } }
  };
  return o;
}

// strided spatial slice: x[:,:, hs::hstep, ws::wstep]. (one Focus patch / ONNX Slice)
inline Tensor slice_hw(const Tensor& x, int64_t hs, int64_t ws, int64_t hstep, int64_t wstep) {
  int64_t N=x->shape[0], C=x->shape[1], H=x->shape[2], W=x->shape[3];
  int64_t OH=(H-hs+hstep-1)/hstep, OW=(W-ws+wstep-1)/wstep;
  auto o=make_tensor({N,C,OH,OW},true);
  for (int64_t n=0;n<N;++n) for (int64_t c=0;c<C;++c) for (int64_t i=0;i<OH;++i) for (int64_t j=0;j<OW;++j)
    o->data[((n*C+c)*OH+i)*OW+j] = x->data[((n*C+c)*H+(hs+i*hstep))*W+(ws+j*wstep)];
  o->parents={x}; Node* op=o.get();
  o->backward_fn=[x,op,N,C,H,W,OH,OW,hs,ws,hstep,wstep]{
    for (int64_t n=0;n<N;++n) for (int64_t c=0;c<C;++c) for (int64_t i=0;i<OH;++i) for (int64_t j=0;j<OW;++j)
      x->grad[((n*C+c)*H+(hs+i*hstep))*W+(ws+j*wstep)] += op->grad[((n*C+c)*OH+i)*OW+j];
  };
  return o;
}

