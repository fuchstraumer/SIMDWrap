#pragma once
#ifndef SIMD_MATH_H
#define SIMD_MATH_H

namespace simd {
#ifdef SIMD_LEVEL_SSE3
	// Elementary math functions using SSE instructions
	#include "SIMD_SSE.h"


	// Take the square root of &in
	__forceinline vec4 vec4::sqrt(vec4 const &in) {
		return vec4(_mm_sqrt_ps(*in.data));
	}

	// Take the inverse square root (1 / square root(x)) of &in
	__forceinline static vec4 invsqrt(vec4 const &in) {
		return vec4(_mm_rsqrt_ps(*in.data));
	}

	// Compare the two inputs and return the maximum in each position
	__forceinline vec4 vec4::max(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_max_ps(*in0.data, *in1.data));
	}

	// Compare the two inputs and return the minimum in each position
	__forceinline vec4 vec4::min(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_min_ps(*in0.data, *in1.data));
	}
	// Convert the input float-vec into an int-vec
	__forceinline static ivec4 ConvertToInt(vec4 const &in) {
		return ivec4(_mm_cvtps_epi32(*in.data));
	}

	// Convert input double vec into an int-vec
	
	// Cast input int-vec into float vec. This is a costless operation
	__forceinline static vec4 CastToFloat(ivec4 const &in) {
		return vec4(_mm_castsi128_ps(*in.data));
	}

	// Convert input int-vec into float vec
	__forceinline static vec4 ConvertToFloat(ivec4 const &in) {
		return vec4(_mm_cvtepi32_ps(*in.data));
	}

	// Cast input float-vec into int-vec
	__forceinline static ivec4 CastToInt(vec4 const &in) {
		return ivec4(_mm_castps_si128(*in.data));
	}

	// Simple linear interpolation
	__forceinline vec4 vec4::lerp(vec4 const &i, vec4 const &j, vec4 const &k) {
		vec4 result;
		result = (j - i);
		result = result + (j * k);
		return result;
	}

	// Normalize the input vector
	__forceinline static vec4 normalize(vec4 const& in) {
		vec4 mul = invsqrt(in);
		vec4 res = mul * in;
		return res;
	}
#endif // SIMD_LEVEL_SSE3

/*
	
	There are some instructions only available with SSE4.1/4.2
	support. These instructions are quite useful, however.

	They are defined below.

*/

#ifdef SIMD_LEVEL_SSE41
	
	// Round the input float-vec down to the nearest integer
	__forceinline static vec4 floor(vec4 const &in) {
		return vec4(_mm_floor_ps(*in.data));
	}
	// Round the input float-vec up to the nearest integer
	__forceinline static vec4 ceil(vec4 const &in) {
		return vec4(_mm_ceil_ps(*in.data));
	}
	// Blend A&B using __m128 mask 
	__forceinline static vec4 blendv(vec4 const &a, vec4 const &b, __m128 const &mask) {
		return vec4(_mm_blendv_ps(*a.data, *b.data, mask));
	}
	__forceinline static vec4 blendv(vec4 const &a, vec4 const &b, vec4 const &mask) {
		return vec4(_mm_blendv_ps(*a.data, *b.data, *mask.data));
	}
#endif // SIMD_LEVEL_SSE41


#ifdef SIMD_LEVEL_AVX2
	#include "SIMD_AVX.h"
	// Elementary math functions using AVX2 instructions
	// Conversion - uses truncation
	__forceinline static ivec8 ConvertToInt(vec8 const& a) {
		return ivec8(_mm256_cvttps_epi32(*a.data));
	}

	// Casting - no overhead, converts directly.
	__forceinline static ivec8 CastToInt(vec8 const& a) {
		return ivec8(_mm256_castps_si256(*a.data));
	}
	
	// Return a vector where each entry is the maximum of the two input vector
	// elements at the same entry position
	__forceinline vec8 vec8::max(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_max_ps(*a.data, *b.data));
	}

	// Return a vector where each entry is the minimum of the two input vector
	// elements at the same entry position
	__forceinline vec8 vec8::min(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_min_ps(*a.data, *b.data));
	}

	// Return a vector that is the square root of the input vector
	__forceinline vec8 vec8::sqrt(vec8 const& a) {
		return vec8(_mm256_sqrt_ps(*a.data));
	}

	// Return a vector that is the inverse (or reciprocal) square root of
	// the input vector
	__forceinline static vec8 invsqrt(vec8 const& a) {
		return vec8(_mm256_rsqrt_ps(*a.data));
	}

	// Return a vector that is the inverse of the input vector
	__forceinline static vec8 inv(vec8 const& a) {
		return vec8(_mm256_rcp_ps(*a.data));
	}

	// Return a vector that is the reciprocal of the input - equivalent to inv()
	__forceinline static vec8 rcp(vec8 const& a) {
		return vec8(_mm256_rcp_ps(*a.data));
	}

	// Floor the input vector (round down to nearest whole number) and return 
	// an integer vector as the result
	__forceinline static ivec8 ifloor(vec8 const& a) {
		return ivec8(CastToInt(_mm256_floor_ps(*a.data)));
	}

	// Floor the input vector (round down to nearest whole number) and return 
	// a float vector as the result
	__forceinline static vec8 floor(vec8 const& a) {
		return vec8(_mm256_floor_ps(*a.data));
	}

	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__forceinline static ivec8 iceil(vec8 const& a) {
		return ivec8(CastToInt(_mm256_ceil_ps(*a.data)));
	}

	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__forceinline static vec8 ceil(vec8 const& a) {
		return vec8(_mm256_ceil_ps(*a.data));
	}
	// Blends between v0 and v1 using the given mask vec
	__forceinline static vec8 blendv(vec8 const& v0, vec8 const& v1, vec8 const& mask) {
		return vec8(_mm256_blendv_ps(*v0.data, *v1.data, *mask.data));
	}
	// takes module of v0 with respect to v1
	__forceinline static vec8 mod(vec8 const& v0, vec8 const& v1) {
		vec8 const divisor0 = v0 / v1;
		vec8 const flr = floor(divisor0);
		vec8 const mul = v1 * flr;
		return vec8(v0 - mul);
	}

	// Various Fused multiply instructions

	// a * b + c
	__forceinline static vec8 fma(vec8 const& a, vec8 const& b, vec8 const& c) {
		vec8 res;
		*res.data = _mm256_fmadd_ps(*a.data, *b.data, *c.data);
		return res;
	}

	// a * b - c
	__forceinline static vec8 fms(vec8 const& a, vec8 const& b, vec8 const& c) {
		vec8 res;
		*res.data = _mm256_fmsub_ps(*a.data, *b.data, *c.data);
		return res;
	}

	// Clamp v0 to the range given by maxval, minval (vector version)
	__forceinline static vec8 clamp(vec8 const& v0, vec8 const& min_val, vec8 const& max_val) {
		vec8 min0 = _mm256_min_ps(*v0.data, *max_val.data);
		vec8 res(_mm256_max_ps(*min0.data, *min_val.data));
		return res;
	}

	// Clamp v0 to the range given by maxval, minval (float version)
	__forceinline static vec8 clamp(vec8 const& v0, const float& min, const float& max) {
		vec8 minv(min), maxv(max);
		vec8 res = clamp(v0, minv, maxv);
		return res;
	}

#endif // SIMD_LEVEL_AVX2


}
#endif // !SIMD_MATH_H
