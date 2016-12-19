#pragma once
#ifndef SIMD_MATH_H
#define SIMD_MATH_H

namespace simd {
#ifdef SIMD_LEVEL_SSE3
	// Elementary math functions using SSE instructions
	#include "SIMD_SSE.h"

	// Return a vector made up of the max of each element from in0 and in1
	__forceinline static ivec4 max(ivec4 const &in0, ivec4 const &in1){
		return ivec4(_mm_max_epi32(in0.Data, in1.Data));
	}
	
	// Take the square root of &in
	__forceinline static vec4 sqrt(vec4 const &in) {
		return vec4(_mm_sqrt_ps(in.Data));
	}

	// Take the inverse square root (1 / square root(x)) of &in
	__forceinline static vec4 invsqrt(vec4 const &in) {
		return vec4(_mm_rsqrt_ps(in.Data));
	}

	// Compare the two inputs and return the maximum in each position
	__forceinline static vec4 max(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_max_ps(in0.Data, in1.Data));
	}

	// Compare the two inputs and return the minimum in each position
	__forceinline static vec4 min(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_min_ps(in0.Data, in1.Data));
	}
	// Convert the input float-vec into an int-vec
	__forceinline static ivec4 ConvertToInt(vec4 const &in) {
		return ivec4(_mm_cvtps_epi32(in.Data));
	}

	// Convert input double vec into an int-vec
	
	// Cast input int-vec into float vec. This is a costless operation
	__forceinline static vec4 CastToFloat(ivec4 const &in) {
		return vec4(_mm_castsi128_ps(in.Data));
	}

	// Convert input int-vec into float vec
	__forceinline static vec4 ConvertToFloat(ivec4 const &in) {
		return vec4(_mm_cvtepi32_ps(in.Data));
	}

	// Cast input float-vec into int-vec
	__forceinline static ivec4 CastToInt(vec4 const &in) {
		return ivec4(_mm_castps_si128(in.Data));
	}

	// Simple linear interpolation
	__forceinline static vec4 lerp(vec4 const &i, vec4 const &j, vec4 const &k) {
		vec4 result;
		result = (j - i);
		result = result + (j * k);
		return result;
	}

	// Normalize the input vector
	__forceinline static vec4 norm(vec4 const& in) {
		vec4 mul = invsqrt(in);
		vec4 res = mul * in;
		return res;
	}

	// Find the exponential pow of the input vector : this is the fast method. Less precise than other method,
	// but yields one scalar result every 3 processor cycles.
	// TODO
	__forceinline static vec4 fastpow() {
		throw("Not implemented");
		return vec4(0.0f);
	}
#endif // SIMD_LEVEL_SSE3

#ifdef SIMD_LEVEL_SSE41
	// Couple extra functions using SSE4.1 instructions
	// Round the input float-vec down to the nearest integer
	__forceinline static vec4 floor(vec4 const &in) {
		return vec4(_mm_floor_ps(in.Data));
	}
	// Round the input float-vec up to the nearest integer
	__forceinline static vec4 ceil(vec4 const &in) {
		return vec4(_mm_ceil_ps(in.Data));
	}
	// Blend A&B using __m128 mask 
	__forceinline static vec4 blendv(vec4 const &a, vec4 const &b, __m128 const &mask) {
		return vec4(_mm_blendv_ps(a.Data, b.Data, mask));
	}
	__forceinline static vec4 blendv(vec4 const &a, vec4 const &b, vec4 const &mask) {
		return vec4(_mm_blendv_ps(a.Data, b.Data, mask.Data));
	}
#endif // SIMD_LEVEL_SSE41

#ifdef SIMD_LEVEL_AVX2

	// Elementary math functions using AVX2 instructions
	// Conversion - uses truncation
	__forceinline static ivec8 ConvertToInt(vec8 const& a) {
		return ivec8(_mm256_cvttps_epi32(a.Data));
	}

	// Casting - no overhead, converts directly.
	__forceinline static ivec8 CastToInt(vec8 const& a) {
		return ivec8(_mm256_castps_si256(a.Data));
	}
	
	// Return a vector where each entry is the maximum of the two input vector
	// elements at the same entry position
	__forceinline static vec8 max(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_max_ps(a.Data, b.Data));
	}

	// Return a vector where each entry is the minimum of the two input vector
	// elements at the same entry position
	__forceinline static vec8 min(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_min_ps(a.Data, b.Data));
	}

	// Return a vector that is the square root of the input vector
	__forceinline static vec8 sqrt(vec8 const& a) {
		return vec8(_mm256_sqrt_ps(a.Data));
	}

	// Return a vector that is the inverse (or reciprocal) square root of
	// the input vector
	__forceinline static vec8 invsqrt(vec8 const& a) {
		return vec8(_mm256_rsqrt_ps(a.Data));
	}

	// Return a vector that is the inverse of the input vector
	__forceinline static vec8 inv(vec8 const& a) {
		return vec8(_mm256_rcp_ps(a.Data));
	}

	// Return a vector that is the reciprocal of the input - equivalent to inv()
	__forceinline static vec8 rcp(vec8 const& a) {
		return vec8(_mm256_rcp_ps(a.Data));
	}

	// Floor the input vector (round down to nearest whole number) and return 
	// an integer vector as the result
	__forceinline static ivec8 ifloor(vec8 const& a) {
		return ivec8(CastToInt(_mm256_floor_ps(a.Data)));
	}

	// Floor the input vector (round down to nearest whole number) and return 
	// a float vector as the result
	__forceinline static vec8 floor(vec8 const& a) {
		return vec8(_mm256_floor_ps(a.Data));
	}

	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__forceinline static ivec8 iceil(vec8 const& a) {
		return ivec8(CastToInt(_mm256_ceil_ps(a.Data)));
	}

	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__forceinline static vec8 ceil(vec8 const& a) {
		return vec8(_mm256_ceil_ps(a.Data));
	}
	// Blends between v0 and v1 using the given mask vec
	__forceinline static vec8 blendv(vec8 const& v0, vec8 const& v1, vec8 const& mask) {
		return vec8(_mm256_blendv_ps(v0.Data, v1.Data, mask.Data));
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
		res.Data = _mm256_fmadd_ps(a.Data, b.Data, c.Data);
		return res;
	}

	// a * b - c
	__forceinline static vec8 fms(vec8 const& a, vec8 const& b, vec8 const& c) {
		vec8 res;
		res.Data = _mm256_fmsub_ps(a.Data, b.Data, c.Data);
		return res;
	}

	// Clamp v0 to the range given by maxval, minval (vector version)
	__forceinline static vec8 clamp(vec8 const& v0, vec8 const& min_val, vec8 const& max_val) {
		vec8 min0 = _mm256_min_ps(v0.Data, max_val.Data);
		vec8 res(_mm256_max_ps(min0.Data, min_val.Data));
		return res;
	}

	// Clamp v0 to the range given by maxval, minval (float version)
	__forceinline static vec8 clam(vec8 const& v0, const float& min, const float& max) {
		vec8 minv(min), maxv(max);
		vec8 res = clamp(v0, minv, maxv);
		return res;
	}

#endif // SIMD_LEVEL_AVX2


}
#endif // !SIMD_MATH_H
