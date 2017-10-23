#pragma once
#ifndef SIMD_GEOMETRIC_H
#define SIMD_GEOMETRIC_H

#include "SIMD.hpp"

#ifdef SIMD_LEVEL_SSE3
namespace simd {
	// Dot product of two input vectors
	__forceinline static vec4 dot(const vec4& v0, const vec4& v1) {
		vec4 mul = v0 * v1;
		vec4 hadd = _mm_hadd_ps(*mul.data, *mul.data);
		return vec4(_mm_hadd_ps(*hadd.data, *hadd.data));
	}
	// Returns vector-length of input vectors
	__forceinline static vec4 length(const vec4& v0) {
		vec4 dot_p = dot(v0, v0);
		return vec4(vec4::sqrt(dot_p));
	}
}

#endif

#ifdef SIMD_LEVEL_AVX2

/*

	Defines common geometric functions for working with AVX2
	intrinsics.

*/

namespace simd {

	// Dot product of two input vectors
	__forceinline static vec8 dot(const vec8& v0, const vec8 &v1) {
		vec8 mul = v0 * v1;
		vec8 hadd0 = _mm256_hadd_ps(*mul.data, *mul.data);
		vec8 res = _mm256_hadd_ps(*hadd0.data, *hadd0.data);
		return res;
	}

	// Cross product of the two input vectors
	__forceinline static vec8 cross(const vec8& v0, const vec8& v1) {

	}

	// Length of this vector, returned as a vector
	__forceinline static vec8 length(const vec8& v0) {
		vec8 prod = dot(v0, v0);
		prod.data = _mm256_sqrt_ps(*prod.data);
		return prod;
	}
}

#endif // SIMD_LEVEL_AVX2

#endif // !SIMD_GEOMETRIC_H
