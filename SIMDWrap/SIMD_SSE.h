#pragma once
#ifndef SIMD_SSE_H
#define SIMD_SSE_H
#include "SIMD.h"
#ifdef SIMD_LEVEL_SSE3
#include <cstdint>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>


namespace simd {
	class ivec4 : public SIMDv<__m128i, 16> {
	public:
		ivec4() : SIMDv() {
			this->Data = _mm_setzero_si128();
		}
		ivec4(int32_t a, int32_t b, int32_t c, int32_t d = 0) {
			this->Data = _mm_set_epi32(a, b, c, d);
		}
		ivec4(int32_t a) {
			this->Data = _mm_set1_epi32(a);
		}
		ivec4(__m128i data) {
			this->Data = data;
		}
		// Store the data from this vector into *data
		void Store(__m128i* data) {
			_mm_store_si128(data, this->Data);
		}
		// Loads 128 bits of data into this->Data
		void Load(__m128i* data) {
			this->Data = _mm_load_si128(data);
		}

		// Basic operations for building this vector
		// Set four separate elements a,b,c,d
		void Set(int32_t a = 0, int32_t b = 0, int32_t c = 0, int32_t d = 0) {
			this->Data = _mm_set_epi32(a, b, c, d);
		}

		// Set all elements to be equal to x
		void Set(int32_t x) {
			this->Data = _mm_set1_epi32(x);
		}

		// Get element at i and return a copy
		int32_t Get(uint8_t index) {
			return (this->Data.m128i_i32[index]);
		}
		// Basic operators
		// Unary operators
		__forceinline ivec4& operator+=(ivec4 const &other) {
			this->Data = _mm_add_epi32(this->Data, other.Data);
			return *this;
		}

		__forceinline  ivec4& operator-=(ivec4 const &other) {
			this->Data = _mm_sub_epi32(this->Data, other.Data);
			return *this;
		}

		__forceinline  ivec4& operator*=(ivec4 const &other) {
			this->Data = _mm_mul_epi32(other.Data, this->Data);
			return *this;
		}

		// Binary operators
		__forceinline ivec4 ivec4::operator+(ivec4 const &other) const {
			return ivec4(_mm_add_epi32(this->Data, other.Data));
		}

		__forceinline ivec4 ivec4::operator-(ivec4 const &other) const {
			return ivec4(_mm_sub_epi32(this->Data, other.Data));
		}

		__forceinline ivec4 operator*(ivec4 const &other) const {
			return ivec4(_mm_mul_epi32(this->Data, other.Data));
		}
		
		__forceinline ivec4 operator<(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(other.Data,this->Data));
		}

		__forceinline ivec4 operator>(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(this->Data, other.Data));
		}

		__forceinline ivec4 operator&(ivec4 const &other) const {
			return ivec4(_mm_and_si128(this->Data, other.Data));
		}

		__forceinline ivec4 operator|(ivec4 const &other) const {
			return ivec4(_mm_or_si128(this->Data, other.Data));
		}

		__forceinline static ivec4 xor(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_xor_si128(in0.Data, in1.Data));
		}

		__forceinline static ivec4 andnot(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_andnot_si128(in0.Data, in0.Data));
		}
		// performing a NOT on this vector is done by xor'ing with a vec
		// set to be 100% 1's
		__forceinline ivec4 operator~() const{
			ivec4 other(0xffffffff);
			return ivec4(_mm_xor_si128(this->Data, other.Data));
		}

		// Bit-shift this vector right using &dist as a mask
		// Fill using zeroes
		__forceinline ivec4 operator >> (ivec4 const& dist) {
			return ivec4(_mm_srl_epi32(this->Data, dist.Data));
		}
		// Bit-shift this vector right by distance dist
		// Fill using zeroes
		__forceinline ivec4 operator >> (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_srl_epi32(this->Data, distv.Data));
		}
		// Shift this vector left using &dist as a mask
		// Fill using zeroes
		__forceinline ivec4 operator << (ivec4 const& dist) {
			return ivec4(_mm_sll_epi32(this->Data, dist.Data));
		}
		// Shfit this vector left by distance dist
		// Fill using zeroes
		__forceinline ivec4 operator << (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_sll_epi32(this->Data, distv.Data));
		}

		__forceinline ivec4 operator==(ivec4 const &other) const {
			return ivec4(_mm_cmpeq_epi32(this->Data,other.Data));
		}

		// More advanced mathematical functions.
		
	};

	class vec4 : public SIMDv<__m128, 16> {
	public:
		// Initialize a vec4 with all elements set to zero
		vec4() {
			this->Data = _mm_setzero_ps();
		}
		// Initialize a vec4 with at least two distinct elements
		vec4(float x, float y, float z = 0.0f, float w = 0.0f) {
			// Order in memory is actually reversed.
			this->Data = _mm_set_ps(w, z, y, x);
		}
		// Initialize a vec4 with all values set to a
		vec4(float a) {
			this->Data = _mm_set1_ps(a);
		}
		// Initialize a vec4 by providing the intrinsic base type
		vec4(__m128 data) {
			this->Data = data;
		}
		// Uniformly set all elements in this vector to a
		void SetOne(float a = 0.0f) {
			this->Data = _mm_set_ps1(a);
		}
		// Store this vec4's data into data
		void Store(__m128 data) {
			// Somehow, just providing a pointer to the first
			// element causes the whole vec4 to be copied over
			_mm_store_ps(&data.m128_f32[0], this->Data);
		}

		// Loads 128 bits of data into this->Data from data
		void Load(__m128 data) {
			__m128 v;
			v = _mm_load_ps(&data.m128_f32[0]);
			this->Data = v;
		}
		// Basic operators

		// Unary operators
		__forceinline vec4& operator+=(vec4 const & other) {
			this->Data = _mm_add_ps(this->Data, other.Data);
			return *this;
		}
		__forceinline vec4& operator-=(vec4 const & other) {
			this->Data = _mm_add_ps(this->Data, other.Data);
			return *this;
		}
		__forceinline vec4& operator*=(vec4 const & other) {
			this->Data = _mm_mul_ps(this->Data, other.Data);
			return *this;
		}
		__forceinline vec4& operator/=(vec4 const & other) {
			this->Data = _mm_div_ps(this->Data, other.Data);
			return *this;
		}
		__forceinline vec4& operator++() {
			vec4 increment; increment.SetOne(1.0f);
			this->Data = _mm_add_ps(this->Data, increment.Data);
			return *this;
		}
		__forceinline vec4& operator--() {
			vec4 decrement; decrement.SetOne(1.0f);
			this->Data = _mm_add_ps(this->Data, decrement.Data);
			return *this;
		}

		// Binary operators
		__forceinline vec4 operator+(vec4 const &other) const {
			return vec4(_mm_add_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator-(vec4 const &other) const {
			return vec4(_mm_sub_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator*(vec4 const &other) const {
			return vec4(_mm_mul_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator/(vec4 const &other) const {
			return vec4(_mm_div_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator<(vec4 const &other) const {
			return vec4(_mm_cmplt_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator>(vec4 const &other) const {
			return vec4(_mm_cmpgt_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator<=(vec4 const &other) const {
			return vec4(_mm_cmple_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator>=(vec4 const &other) const {
			return vec4(_mm_cmpge_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator&(vec4 const &other) const {
			return vec4(_mm_and_ps(this->Data, other.Data));
		}
		__forceinline vec4 operator|(vec4 const& other) const {
			return vec4(_mm_or_ps(this->Data, other.Data));
		}
		// xor this vector with another vector
		__forceinline vec4 xor(vec4 const& other) const {
			return vec4(_mm_xor_ps(this->Data, other.Data));
		}
		
		// static functions (for operation on two distinct vectors)

		__forceinline static vec4 xor(vec4 const& in0, vec4 const& in1) {
			return vec4(_mm_xor_ps(in0.Data, in1.Data));
		}
		__forceinline vec4 andnot(vec4 const& other) const {
			return vec4(_mm_andnot_ps(this->Data, other.Data));
		}
		
		__forceinline static vec4 dot(vec4 const& v0, vec4 const& v1) {
			vec4 mul = v0 * v1;
			vec4 tmp = _mm_shuffle_ps(mul.Data, mul.Data, _MM_SHUFFLE(3, 2, 1, 0));
			tmp += mul;
			vec4 res = _mm_shuffle_ps(tmp.Data, tmp.Data, _MM_SHUFFLE(2, 3, 0, 1));
			res += tmp;
			return res;
		}

		/*
			Following defined in SIMD_Math.h
		*/

		// Takes the sqrt of each element in the given input vector
		static vec4 sqrt(vec4 const & in);

		// Returns the maximum of the two input vectors in each element
		// (i.e, a vector containing the max of both vector's elements)
		static vec4 max(vec4 const & in0, vec4 const & in1);

		// Returns the minimum of the two input vectors in each element
		// (i.e, a vector containing the mins of both vector's elements)
		static vec4 min(vec4 const & in0, vec4 const & in1);

		// Linearly interpolates between the input vectors
		static vec4 lerp(vec4 const & i, vec4 const & j, vec4 const & k);
};
}
#endif

#endif // !SIMD_SSE_H
