#pragma once
#ifndef SIMD_H
#define SIMD_H
// TODO: Detect max supported level automatically.
#define SIMD_LEVEL_SSE3
#define SIMD_LEVEL_SSE41
// Currently only using SSE3 + one SSE41 function
#define COMPILER_MSVC
#if defined(COMPILER_MSVC) || defined (COMPILER_INTEL)
// TODO: set aligned alloc and aligned free macros here
#elif defined(COMPILER_GNU) || defined(COMPILER_CLANG)
// TODO: set aligned alloc and aligned free macros here
#endif
// AVX implementation not functional. AVX is very sparse
// features-wise, AVX2 has actual features
#ifdef SIMD_LEVEL_AVX
#include "SIMD_AVX.h"
#endif // SIMD_LEVEL_AVX
#ifdef SIMD_LEVEL_SSE3
#include "SIMD_SSE.h"
#endif


#endif // !SIMD_H
