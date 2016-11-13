#pragma once
#ifndef SIMD_H
#define SIMD_H
#define SIMD_LEVEL_SSE3
#define SIMD_LEVEL_SSE41
// Currently only using AVX
#define COMPILER_MSVC
#if defined(COMPILER_MSVC) || defined (COMPILER_INTEL)
#define BEGIN_ALIGNED(n,a) __declspec(align(a)) n; //
#define END_ALIGNED(n,a)
#elif defined(COMPILER_GNU) || defined(COMPILER_CLANG)
#define BEGIN_ALIGNED(n,a) n
#define END_ALIGNED(n,a) __attribute__(aligned(a))
#endif

#ifdef SIMD_LEVEL_AVX
#include "SIMD_AVX.h"
#endif // SIMD_LEVEL_AVX
#ifdef SIMD_LEVEL_SSE3
#include "SIMD_SSE.h"
#endif


#endif // !SIMD_H
