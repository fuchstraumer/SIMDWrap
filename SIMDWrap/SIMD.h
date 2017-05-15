#pragma once
#ifndef SIMD_H
#define SIMD_H

#include "SIMDv.h"
#if defined(_WIN64)
// 64-bit CPU's should all support SSE3 instruction set
#define SIMD_LEVEL_SSE3
#endif

#define SIMD_LEVEL_AVX2
#define SIMD_LEVEL_SSE41
#define SIMD_LEVEL_SSE3

#ifdef SIMD_LEVEL_AVX2
#include "SIMD_AVX.h"
#endif // SIMD_LEVEL_AVX
#ifdef SIMD_LEVEL_SSE3
#include "SIMD_SSE.h"
#endif


#endif // !SIMD_H
