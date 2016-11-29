#pragma once
#ifndef SIMD_MATH_H
#define SIMD_MATH_H

namespace simd {
#ifdef SIMD_LEVEL_SSE3
	// Elementary math functions using SSE instructions
	#include "SIMD_SSE.h"
	// Special functions for vectors
		// Ivec funcs
	__inline static ivec4 max(ivec4 const &in0, ivec4 const &in1){
		return ivec4(_mm_max_epi32(in0.Data, in1.Data));
	}
		// vec funcs
	// Take the square root of &in
	__inline static vec4 const sqrt(vec4 const &in) {
		return vec4(_mm_sqrt_ps(in.Data));
	}
	// Take the inverse square root (1 / square root(x)) of &in
	__inline static vec4 const invsqrt(vec4 const &in) {
		return vec4(_mm_rsqrt_ps(in.Data));
	}
	// Compare the two inputs and return the maximum in each position
	__inline static vec4 const max(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_max_ps(in0.Data, in1.Data));
	}
	// Compare the two inputs and return the minimum in each position
	__inline static vec4 const min(vec4 const &in0, vec4 const &in1) {
		return vec4(_mm_min_ps(in0.Data, in1.Data));
	}
	// Convert the input float-vec into an int-vec
	__inline static ivec4 const ConvertToInt(vec4 const &in) {
		return ivec4(_mm_cvtps_epi32(in.Data));
	}
	// Convert input double vec into an int-vec
	
	// Cast input int-vec into float vec. This is a costless operation
	__inline static vec4 const CastToFloat(ivec4 const &in) {
		return vec4(_mm_castsi128_ps(in.Data));
	}
	// Convert input int-vec into float vec
	__inline static vec4 ConvertToFloat(ivec4 const &in) {
		return vec4(_mm_cvtepi32_ps(in.Data));
	}
	// Cast input float-vec into int-vec
	__inline static ivec4 CastToInt(vec4 const &in) {
		return ivec4(_mm_castps_si128(in.Data));
	}
	// Simple linear interpolation
	__inline static vec4 lerp(vec4 const &i, vec4 const &j, vec4 const &k) {
		vec4 result;
		result = (j - i);
		result = result + (j * k);
		return result;
	}
#endif // SIMD_LEVEL_SSE3

#ifdef SIMD_LEVEL_SSE41
	// Couple extra functions using SSE4.1 instructions
	// Round the input float-vec down to the nearest integer
	__inline static vec4 const floor(vec4 const &in) {
		return vec4(_mm_floor_ps(in.Data));
	}
	// Round the input float-vec up to the nearest integer
	__inline static vec4 const ceil(vec4 const &in) {
		return vec4(_mm_ceil_ps(in.Data));
	}
	// Blend A&B using __m128 mask 
	__inline static vec4 blendv(vec4 const &a, vec4 const &b, __m128 const &mask) {
		return vec4(_mm_blendv_ps(a.Data, b.Data, mask));
	}
	__inline static vec4 blendv(vec4 const &a, vec4 const &b, vec4 const &mask) {
		return vec4(_mm_blendv_ps(a.Data, b.Data, mask.Data));
	}
#endif // SIMD_LEVEL_SSE41

#ifdef SIMD_LEVEL_AVX2
	// Elementary math functions using AVX2 instructions
	
	// Return a vector where each entry is the maximum of the two input vector
	// elements at the same entry position
	__inline static vec8 max(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_max_ps(a.Data, b.Data));
	}
	// Return a vector where each entry is the minimum of the two input vector
	// elements at the same entry position
	__inline static vec8 min(vec8 const& a, vec8 const& b) {
		return vec8(_mm256_min_ps(a.Data, b.Data));
	}
	// Return a vector that is the square root of the input vector
	__inline static vec8 sqrt(vec8 const& a) {
		return vec8(_mm256_sqrt_ps(a.Data));
	}
	// Return a vector that is the inverse (or reciprocal) square root of
	// the input vector
	__inline static vec8 invsqrt(vec8 const& a) {
		return vec8(_mm256_rsqrt_ps(a.Data));
	}
	// Return a vector that is the inverse of the input vector
	__inline static vec8 inv(vec8 const& a) {
		return vec8(_mm256_rcp_ps(a.Data));
	}
	// Return a vector that is the reciprocal of the input - equivalent to inv()
	__inline static vec8 rcp(vec8 const& a) {
		return vec8(_mm256_rcp_ps(a.Data));
	}
	// Floor the input vector (round down to nearest whole number) and return 
	// an integer vector as the result
	__inline static ivec8 const ifloor(vec8 const& a) {
		return ivec8(vec8::CastToInt(_mm256_floor_ps(a.Data)));
	}
	// Floor the input vector (round down to nearest whole number) and return 
	// a float vector as the result
	__inline static vec8 const floor(vec8 const& a) {
		return vec8(_mm256_floor_ps(a.Data));
	}
	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__inline static ivec8 const iceil(vec8 const& a) {
		return ivec8(vec8::CastToInt(_mm256_ceil_ps(a.Data)));
	}
	// Ceil the input vector (round up to nearest whole number and return
	// an integer vector as the result
	__inline static vec8 const ceil(vec8 const& a) {
		return vec8(_mm256_ceil_ps(a.Data));
	}

#endif // SIMD_LEVEL_AVX2


}
#endif // !SIMD_MATH_H
