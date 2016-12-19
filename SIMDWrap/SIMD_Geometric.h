#pragma once
#ifndef SIMD_GEOMETRIC_H
#define SIMD_GEOMETRIC_H

#include "SIMD.h"

#ifdef SIMD_LEVEL_AVX2

/*

	Defines common geometric functions for working with AVX2
	intrinsics.

*/

namespace simd {

	// Dot product of two input vectors
	__forceinline static vec8 dot(const vec8& v0, const vec8 &v1) {
		vec8 mul = v0 * v1;
		vec8 hadd0 = _mm256_hadd_ps(mul.Data, mul.Data);
		vec8 res = _mm256_hadd_ps(hadd0.Data, hadd0.Data);
		return res;
	}

	// Cross product of the two input vectors
	__forceinline static vec8 cross(const vec8& v0, const vec8& v1) {

	}

	// Length of this vector, returned as a vector
	__forceinline static vec8 length(const vec8& v0) {
		vec8 prod = dot(v0, v0);
		prod.Data = _mm256_sqrt_ps(prod.Data);
		return prod;
	}
}

#endif // SIMD_LEVEL_AVX2

#endif // !SIMD_GEOMETRIC_H
