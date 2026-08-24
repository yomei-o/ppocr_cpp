// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_H
#define EIGEN_CORE_H

// first thing Eigen does: stop the compiler from reporting useless warnings.
#include "Eigen_src_Core_util_DisableStupidWarnings.h"

// then include this file where all our macros are defined. It's really important to do it first because
// it's where we do all the compiler/OS/arch detections and define most defaults.
#include "Eigen_src_Core_util_Macros.h"

// This detects SSE/AVX/NEON/etc. and configure alignment settings
#include "Eigen_src_Core_util_ConfigureVectorization.h"

// We need cuda_runtime.h/hip_runtime.h to ensure that
// the EIGEN_USING_STD macro works properly on the device side
#if defined(EIGEN_CUDACC)
  #include <cuda_runtime.h>
#elif defined(EIGEN_HIPCC)
  #include <hip/hip_runtime.h>
#endif


#ifdef EIGEN_EXCEPTIONS
  #include <new>
#endif

// Disable the ipa-cp-clone optimization flag with MinGW 6.x or newer (enabled by default with -O3)
// See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556 for details.
#if EIGEN_COMP_MINGW && EIGEN_GNUC_AT_LEAST(4,6) && EIGEN_GNUC_AT_MOST(5,5)
  #pragma GCC optimize ("-fno-ipa-cp-clone")
#endif

// Prevent ICC from specializing std::complex operators that silently fail
// on device. This allows us to use our own device-compatible specializations
// instead.
#if defined(EIGEN_COMP_ICC) && defined(EIGEN_GPU_COMPILE_PHASE) \
    && !defined(_OVERRIDE_COMPLEX_SPECIALIZATION_)
#define _OVERRIDE_COMPLEX_SPECIALIZATION_ 1
#endif
#include <complex>

// this include file manages BLAS and MKL related macros
// and inclusion of their respective header files
#include "Eigen_src_Core_util_MKL_support.h"


#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
  #define EIGEN_HAS_GPU_FP16
#endif

#if defined(EIGEN_HAS_CUDA_BF16) || defined(EIGEN_HAS_HIP_BF16)
  #define EIGEN_HAS_GPU_BF16
#endif

#if (defined _OPENMP) && (!defined EIGEN_DONT_PARALLELIZE)
  #define EIGEN_HAS_OPENMP
#endif

#ifdef EIGEN_HAS_OPENMP
#include <omp.h>
#endif

// MSVC for windows mobile does not have the errno.h file
#if !(EIGEN_COMP_MSVC && EIGEN_OS_WINCE) && !EIGEN_COMP_ARM
#define EIGEN_HAS_ERRNO
#endif

#ifdef EIGEN_HAS_ERRNO
#include <cerrno>
#endif
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <functional>
#include <sstream>
#ifndef EIGEN_NO_IO
  #include <iosfwd>
#endif
#include <cstring>
#include <string>
#include <limits>
#include <climits> // for CHAR_BIT
// for min/max:
#include <algorithm>

#if EIGEN_HAS_CXX11
#include <array>
#endif

// for std::is_nothrow_move_assignable
#ifdef EIGEN_INCLUDE_TYPE_TRAITS
#include <type_traits>
#endif

// for outputting debug info
#ifdef EIGEN_DEBUG_ASSIGN
#include <iostream>
#endif

// required for __cpuid, needs to be included after cmath
#if EIGEN_COMP_MSVC && EIGEN_ARCH_i386_OR_x86_64 && !EIGEN_OS_WINCE
  #include <intrin.h>
#endif

#if defined(EIGEN_USE_SYCL)
  #undef min
  #undef max
  #undef isnan
  #undef isinf
  #undef isfinite
  #include <CL/sycl.hpp>
  #include <map>
  #include <memory>
  #include <utility>
  #include <thread>
  #ifndef EIGEN_SYCL_LOCAL_THREAD_DIM0
  #define EIGEN_SYCL_LOCAL_THREAD_DIM0 16
  #endif
  #ifndef EIGEN_SYCL_LOCAL_THREAD_DIM1
  #define EIGEN_SYCL_LOCAL_THREAD_DIM1 16
  #endif
#endif


#if defined EIGEN2_SUPPORT_STAGE40_FULL_EIGEN3_STRICTNESS || defined EIGEN2_SUPPORT_STAGE30_FULL_EIGEN3_API || defined EIGEN2_SUPPORT_STAGE20_RESOLVE_API_CONFLICTS || defined EIGEN2_SUPPORT_STAGE10_FULL_EIGEN2_API || defined EIGEN2_SUPPORT
// This will generate an error message:
#error Eigen2-support is only available up to version 3.2. Please go to "http://eigen.tuxfamily.org/index.php?title=Eigen2" for further information
#endif

namespace Eigen {

// we use size_t frequently and we'll never remember to prepend it with std:: every time just to
// ensure QNX/QCC support
using std::size_t;
// gcc 4.6.0 wants std:: for ptrdiff_t
using std::ptrdiff_t;

}

/** \defgroup Core_Module Core module
  * This is the main module of Eigen providing dense matrix and vector support
  * (both fixed and dynamic size) with all the features corresponding to a BLAS library
  * and much more...
  *
  * \code
  * #include "Eigen_Core.h"
  * \endcode
  */

#include "Eigen_src_Core_util_Constants.h"
#include "Eigen_src_Core_util_Meta.h"
#include "Eigen_src_Core_util_ForwardDeclarations.h"
#include "Eigen_src_Core_util_StaticAssert.h"
#include "Eigen_src_Core_util_XprHelper.h"
#include "Eigen_src_Core_util_Memory.h"
#include "Eigen_src_Core_util_IntegralConstant.h"
#include "Eigen_src_Core_util_SymbolicIndex.h"

#include "Eigen_src_Core_NumTraits.h"
#include "Eigen_src_Core_MathFunctions.h"
#include "Eigen_src_Core_GenericPacketMath.h"
#include "Eigen_src_Core_MathFunctionsImpl.h"
#include "Eigen_src_Core_arch_Default_ConjHelper.h"
// Generic half float support
#include "Eigen_src_Core_arch_Default_Half.h"
#include "Eigen_src_Core_arch_Default_BFloat16.h"
#include "Eigen_src_Core_arch_Default_TypeCasting.h"
#include "Eigen_src_Core_arch_Default_GenericPacketMathFunctionsFwd.h"

#if defined EIGEN_VECTORIZE_AVX512
  #include "Eigen_src_Core_arch_SSE_PacketMath.h"
  #include "Eigen_src_Core_arch_SSE_TypeCasting.h"
  #include "Eigen_src_Core_arch_SSE_Complex.h"
  #include "Eigen_src_Core_arch_AVX_PacketMath.h"
  #include "Eigen_src_Core_arch_AVX_TypeCasting.h"
  #include "Eigen_src_Core_arch_AVX_Complex.h"
  #include "src/Core/arch/AVX512/PacketMath.h"
  #include "src/Core/arch/AVX512/TypeCasting.h"
  #include "src/Core/arch/AVX512/Complex.h"
  #include "Eigen_src_Core_arch_SSE_MathFunctions.h"
  #include "Eigen_src_Core_arch_AVX_MathFunctions.h"
  #include "src/Core/arch/AVX512/MathFunctions.h"
#elif defined EIGEN_VECTORIZE_AVX
  // Use AVX for floats and doubles, SSE for integers
  #include "Eigen_src_Core_arch_SSE_PacketMath.h"
  #include "Eigen_src_Core_arch_SSE_TypeCasting.h"
  #include "Eigen_src_Core_arch_SSE_Complex.h"
  #include "Eigen_src_Core_arch_AVX_PacketMath.h"
  #include "Eigen_src_Core_arch_AVX_TypeCasting.h"
  #include "Eigen_src_Core_arch_AVX_Complex.h"
  #include "Eigen_src_Core_arch_SSE_MathFunctions.h"
  #include "Eigen_src_Core_arch_AVX_MathFunctions.h"
#elif defined EIGEN_VECTORIZE_SSE
  #include "Eigen_src_Core_arch_SSE_PacketMath.h"
  #include "Eigen_src_Core_arch_SSE_TypeCasting.h"
  #include "Eigen_src_Core_arch_SSE_MathFunctions.h"
  #include "Eigen_src_Core_arch_SSE_Complex.h"
#elif defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
  #include "src/Core/arch/AltiVec/PacketMath.h"
  #include "src/Core/arch/AltiVec/MathFunctions.h"
  #include "src/Core/arch/AltiVec/Complex.h"
#elif defined EIGEN_VECTORIZE_NEON
  #include "src/Core/arch/NEON/PacketMath.h"
  #include "src/Core/arch/NEON/TypeCasting.h"
  #include "src/Core/arch/NEON/MathFunctions.h"
  #include "src/Core/arch/NEON/Complex.h"
#elif defined EIGEN_VECTORIZE_SVE
  #include "src/Core/arch/SVE/PacketMath.h"
  #include "src/Core/arch/SVE/TypeCasting.h"
  #include "src/Core/arch/SVE/MathFunctions.h"
#elif defined EIGEN_VECTORIZE_ZVECTOR
  #include "src/Core/arch/ZVector/PacketMath.h"
  #include "src/Core/arch/ZVector/MathFunctions.h"
  #include "src/Core/arch/ZVector/Complex.h"
#elif defined EIGEN_VECTORIZE_MSA
  #include "src/Core/arch/MSA/PacketMath.h"
  #include "src/Core/arch/MSA/MathFunctions.h"
  #include "src/Core/arch/MSA/Complex.h"
#endif

#if defined EIGEN_VECTORIZE_GPU
  #include "src/Core/arch/GPU/PacketMath.h"
  #include "src/Core/arch/GPU/MathFunctions.h"
  #include "src/Core/arch/GPU/TypeCasting.h"
#endif

#if defined(EIGEN_USE_SYCL)
  #include "src/Core/arch/SYCL/SyclMemoryModel.h"
  #include "src/Core/arch/SYCL/InteropHeaders.h"
#if !defined(EIGEN_DONT_VECTORIZE_SYCL)
  #include "src/Core/arch/SYCL/PacketMath.h"
  #include "src/Core/arch/SYCL/MathFunctions.h"
  #include "src/Core/arch/SYCL/TypeCasting.h"
#endif
#endif

#include "Eigen_src_Core_arch_Default_Settings.h"
// This file provides generic implementations valid for scalar as well
#include "Eigen_src_Core_arch_Default_GenericPacketMathFunctions.h"

#include "Eigen_src_Core_functors_TernaryFunctors.h"
#include "Eigen_src_Core_functors_BinaryFunctors.h"
#include "Eigen_src_Core_functors_UnaryFunctors.h"
#include "Eigen_src_Core_functors_NullaryFunctors.h"
#include "Eigen_src_Core_functors_StlFunctors.h"
#include "Eigen_src_Core_functors_AssignmentFunctors.h"

// Specialized functors to enable the processing of complex numbers
// on CUDA devices
#ifdef EIGEN_CUDACC
#include "src/Core/arch/CUDA/Complex.h"
#endif

#include "Eigen_src_Core_util_IndexedViewHelper.h"
#include "Eigen_src_Core_util_ReshapedHelper.h"
#include "Eigen_src_Core_ArithmeticSequence.h"
#ifndef EIGEN_NO_IO
  #include "Eigen_src_Core_IO.h"
#endif
#include "Eigen_src_Core_DenseCoeffsBase.h"
#include "Eigen_src_Core_DenseBase.h"
#include "Eigen_src_Core_MatrixBase.h"
#include "Eigen_src_Core_EigenBase.h"

#include "Eigen_src_Core_Product.h"
#include "Eigen_src_Core_CoreEvaluators.h"
#include "Eigen_src_Core_AssignEvaluator.h"

#ifndef EIGEN_PARSED_BY_DOXYGEN // work around Doxygen bug triggered by Assign.h r814874
                                // at least confirmed with Doxygen 1.5.5 and 1.5.6
  #include "Eigen_src_Core_Assign.h"
#endif

#include "Eigen_src_Core_ArrayBase.h"
#include "Eigen_src_Core_util_BlasUtil.h"
#include "Eigen_src_Core_DenseStorage.h"
#include "Eigen_src_Core_NestByValue.h"

// #include "src/Core/ForceAlignedAccess.h"

#include "Eigen_src_Core_ReturnByValue.h"
#include "Eigen_src_Core_NoAlias.h"
#include "Eigen_src_Core_PlainObjectBase.h"
#include "Eigen_src_Core_Matrix.h"
#include "Eigen_src_Core_Array.h"
#include "Eigen_src_Core_CwiseTernaryOp.h"
#include "Eigen_src_Core_CwiseBinaryOp.h"
#include "Eigen_src_Core_CwiseUnaryOp.h"
#include "Eigen_src_Core_CwiseNullaryOp.h"
#include "Eigen_src_Core_CwiseUnaryView.h"
#include "Eigen_src_Core_SelfCwiseBinaryOp.h"
#include "Eigen_src_Core_Dot.h"
#include "Eigen_src_Core_StableNorm.h"
#include "Eigen_src_Core_Stride.h"
#include "Eigen_src_Core_MapBase.h"
#include "Eigen_src_Core_Map.h"
#include "Eigen_src_Core_Ref.h"
#include "Eigen_src_Core_Block.h"
#include "Eigen_src_Core_VectorBlock.h"
#include "Eigen_src_Core_IndexedView.h"
#include "Eigen_src_Core_Reshaped.h"
#include "Eigen_src_Core_Transpose.h"
#include "Eigen_src_Core_DiagonalMatrix.h"
#include "Eigen_src_Core_Diagonal.h"
#include "Eigen_src_Core_DiagonalProduct.h"
#include "Eigen_src_Core_Redux.h"
#include "Eigen_src_Core_Visitor.h"
#include "Eigen_src_Core_Fuzzy.h"
#include "Eigen_src_Core_Swap.h"
#include "Eigen_src_Core_CommaInitializer.h"
#include "Eigen_src_Core_GeneralProduct.h"
#include "Eigen_src_Core_Solve.h"
#include "Eigen_src_Core_Inverse.h"
#include "Eigen_src_Core_SolverBase.h"
#include "Eigen_src_Core_PermutationMatrix.h"
#include "Eigen_src_Core_Transpositions.h"
#include "Eigen_src_Core_TriangularMatrix.h"
#include "Eigen_src_Core_SelfAdjointView.h"
#include "Eigen_src_Core_products_GeneralBlockPanelKernel.h"
#include "Eigen_src_Core_products_Parallelizer.h"
#include "Eigen_src_Core_ProductEvaluators.h"
#include "Eigen_src_Core_products_GeneralMatrixVector.h"
#include "Eigen_src_Core_products_GeneralMatrixMatrix.h"
#include "Eigen_src_Core_SolveTriangular.h"
#include "Eigen_src_Core_products_GeneralMatrixMatrixTriangular.h"
#include "Eigen_src_Core_products_SelfadjointMatrixVector.h"
#include "Eigen_src_Core_products_SelfadjointMatrixMatrix.h"
#include "Eigen_src_Core_products_SelfadjointProduct.h"
#include "Eigen_src_Core_products_SelfadjointRank2Update.h"
#include "Eigen_src_Core_products_TriangularMatrixVector.h"
#include "Eigen_src_Core_products_TriangularMatrixMatrix.h"
#include "Eigen_src_Core_products_TriangularSolverMatrix.h"
#include "Eigen_src_Core_products_TriangularSolverVector.h"
#include "Eigen_src_Core_BandMatrix.h"
#include "Eigen_src_Core_CoreIterators.h"
#include "Eigen_src_Core_ConditionEstimator.h"

#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
  #include "src/Core/arch/AltiVec/MatrixProduct.h"
#elif defined EIGEN_VECTORIZE_NEON
  #include "src/Core/arch/NEON/GeneralBlockPanelKernel.h"
#endif

#include "Eigen_src_Core_BooleanRedux.h"
#include "Eigen_src_Core_Select.h"
#include "Eigen_src_Core_VectorwiseOp.h"
#include "Eigen_src_Core_PartialReduxEvaluator.h"
#include "Eigen_src_Core_Random.h"
#include "Eigen_src_Core_Replicate.h"
#include "Eigen_src_Core_Reverse.h"
#include "Eigen_src_Core_ArrayWrapper.h"
#include "Eigen_src_Core_StlIterators.h"

#ifdef EIGEN_USE_BLAS
#include "src/Core/products/GeneralMatrixMatrix_BLAS.h"
#include "src/Core/products/GeneralMatrixVector_BLAS.h"
#include "src/Core/products/GeneralMatrixMatrixTriangular_BLAS.h"
#include "src/Core/products/SelfadjointMatrixMatrix_BLAS.h"
#include "src/Core/products/SelfadjointMatrixVector_BLAS.h"
#include "src/Core/products/TriangularMatrixMatrix_BLAS.h"
#include "src/Core/products/TriangularMatrixVector_BLAS.h"
#include "src/Core/products/TriangularSolverMatrix_BLAS.h"
#endif // EIGEN_USE_BLAS

#ifdef EIGEN_USE_MKL_VML
#include "src/Core/Assign_MKL.h"
#endif

#include "Eigen_src_Core_GlobalFunctions.h"

#include "Eigen_src_Core_util_ReenableStupidWarnings.h"

#endif // EIGEN_CORE_H
