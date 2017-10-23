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
			*data = _mm_setzero_si128();
		}

		ivec4(const int32_t& a, const int32_t& b, const int32_t& c, const int32_t& d = 0) {
			*data = _mm_set_epi32(a, b, c, d);
		}

		ivec4(const int32_t& a) {
			*data = _mm_set1_epi32(a);
		}

		ivec4(__m128i _data) {
			*data = std::move(_data);
		}

		// Store the data from this vector into *data
		void Store(__m128i* data) {
			_mm_store_si128(data, *data);
		}
		// Loads 128 bits of data into *data
		void Load(__m128i* data) {
			*data = _mm_load_si128(data);
		}

		// Basic operations for building this vector
		// Set four separate elements a,b,c,d
		void Set(const int32_t& a = 0, const int32_t& b = 0, const int32_t& c = 0, const int32_t& d = 0) {
			*data = _mm_set_epi32(a, b, c, d);
		}

		// Set all elements to be equal to x
		void Set(const int32_t& x) {
			*data = _mm_set1_epi32(x);
		}

		// Basic operators
		// Unary operators
		__forceinline ivec4& operator+=(ivec4 const &other) {
			*data = _mm_add_epi32(*data, *other.data);
			return *this;
		}

		__forceinline  ivec4& operator-=(ivec4 const &other) {
			*data = _mm_sub_epi32(*data, *other.data);
			return *this;
		}

		__forceinline  ivec4& operator*=(ivec4 const &other) {
			*data = _mm_mul_epi32(*other.data, *data);
			return *this;
		}

		// Binary operators
		__forceinline ivec4 ivec4::operator+(ivec4 const &other) const {
			return ivec4(_mm_add_epi32(*data, *other.data));
		}

		__forceinline ivec4 ivec4::operator-(ivec4 const &other) const {
			return ivec4(_mm_sub_epi32(*data, *other.data));
		}

		__forceinline ivec4 operator*(ivec4 const &other) const {
			return ivec4(_mm_mul_epi32(*data, *other.data));
		}
		
		__forceinline ivec4 operator<(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(*other.data,*data));
		}

		__forceinline ivec4 operator>(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(*data, *other.data));
		}

		__forceinline ivec4 operator&(ivec4 const &other) const {
			return ivec4(_mm_and_si128(*data, *other.data));
		}

		__forceinline ivec4 operator|(ivec4 const &other) const {
			return ivec4(_mm_or_si128(*data, *other.data));
		}

		__forceinline static ivec4 xor(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_xor_si128(*in0.data, *in1.data));
		}

		__forceinline static ivec4 andnot(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_andnot_si128(*in0.data, *in0.data));
		}

		// performing a NOT on this vector is done by xor'ing with a vec
		// set to be 100% 1's
		__forceinline ivec4 operator~() const{
			ivec4 other(0xffffffff);
			return ivec4(_mm_xor_si128(*data, *other.data));
		}

		// Bit-shift this vector right using &dist as a mask
		// Fill using zeroes
		__forceinline ivec4 operator >> (ivec4 const& dist) {
			return ivec4(_mm_srl_epi32(*data, *dist.data));
		}
		// Bit-shift this vector right by distance dist
		// Fill using zeroes
		__forceinline ivec4 operator >> (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_srl_epi32(*data, *distv.data));
		}
		// Shift this vector left using &dist as a mask
		// Fill using zeroes
		__forceinline ivec4 operator << (ivec4 const& dist) {
			return ivec4(_mm_sll_epi32(*data, *dist.data));
		}
		// Shfit this vector left by distance dist
		// Fill using zeroes
		__forceinline ivec4 operator << (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_sll_epi32(*data, *distv.data));
		}

		__forceinline ivec4 operator==(ivec4 const &other) const {
			return ivec4(_mm_cmpeq_epi32(*data,*other.data));
		}

		// More advanced mathematical functions.
		
	};

	class vec4 : public SIMDv<__m128, 16> {
	public:

		// Initialize a vec4 with all elements set to zero
		vec4() : SIMDv() {
			*data = _mm_setzero_ps();
		}

		// Initialize a vec4 with at least two distinct elements
		vec4(const float& x, const float& y, const float& z = 0.0f, const float& w = 0.0f) : SIMDv() {
			// Order in memory is actually reversed.
			*data = _mm_set_ps(w, z, y, x);
		}

		// Initialize a vec4 with all values set to a
		vec4(const float& a) : SIMDv() {
			*data = _mm_set1_ps(a);
		}

		// Initialize a vec4 by providing the intrinsic base type
		vec4(__m128 _data) : SIMDv() {
			*data = std::move(_data);
		}

		// Uniformly set all elements in this vector to a
		void Fill(const float& a = 0.0f) {
			*data = _mm_set_ps1(a);
		}

		// Store this vec4's data into data
		void Store(__m128 _data) {
			_mm_store_ps(&data->m128_f32[0], _data);
		}

		// Load specified pointer into data
		void Load(float* ptr) {
			*data = _mm_load_ps(ptr);
		}

		// Basic operators

		// Unary operators
		__forceinline vec4& operator+=(vec4 const & other) {
			*data = _mm_add_ps(*data, *other.data);
			return *this;
		}
		__forceinline vec4& operator-=(vec4 const & other) {
			*data = _mm_add_ps(*data, *other.data);
			return *this;
		}
		__forceinline vec4& operator*=(vec4 const & other) {
			*data = _mm_mul_ps(*data, *other.data);
			return *this;
		}
		__forceinline vec4& operator/=(vec4 const & other) {
			*data = _mm_div_ps(*data, *other.data);
			return *this;
		}

		// pre & post increment/decrement operators

		__forceinline vec4& operator++() {
			static const __m128 one = _mm_set1_ps(1.0f);
			*data = _mm_add_ps(*data, one);
			return *this;
		}

		__forceinline vec4& operator--() {
			static const __m128 one = _mm_set1_ps(1.0f);
			*data = _mm_sub_ps(*data, one);
			return *this;
		}

		__forceinline vec4 operator++(int) {
			static const __m128 one = _mm_set1_ps(1.0f);
			vec4 result = *this;
			*data = _mm_add_ps(*data, one);
			return result;
		}

		__forceinline vec4 operator--(int) {
			static const __m128 one = _mm_set1_ps(1.0f);
			vec4 result = *this;
			*data = _mm_sub_ps(*data, one);
			return result;
		}

		// Binary operators
		__forceinline vec4 operator+(vec4 const &other) const {
			return vec4(_mm_add_ps(*data, *other.data));
		}
		__forceinline vec4 operator-(vec4 const &other) const {
			return vec4(_mm_sub_ps(*data, *other.data));
		}
		__forceinline vec4 operator*(vec4 const &other) const {
			return vec4(_mm_mul_ps(*data, *other.data));
		}
		__forceinline vec4 operator/(vec4 const &other) const {
			return vec4(_mm_div_ps(*data, *other.data));
		}
		__forceinline vec4 operator<(vec4 const &other) const {
			return vec4(_mm_cmplt_ps(*data, *other.data));
		}
		__forceinline vec4 operator>(vec4 const &other) const {
			return vec4(_mm_cmpgt_ps(*data, *other.data));
		}
		__forceinline vec4 operator<=(vec4 const &other) const {
			return vec4(_mm_cmple_ps(*data, *other.data));
		}
		__forceinline vec4 operator>=(vec4 const &other) const {
			return vec4(_mm_cmpge_ps(*data, *other.data));
		}
		__forceinline vec4 operator&(vec4 const &other) const {
			return vec4(_mm_and_ps(*data, *other.data));
		}
		__forceinline vec4 operator|(vec4 const& other) const {
			return vec4(_mm_or_ps(*data, *other.data));
		}
		// xor this vector with another vector
		__forceinline vec4 xor(vec4 const& other) const {
			return vec4(_mm_xor_ps(*data, *other.data));
		}
		
		// static functions (for operation on two distinct vectors)

		__forceinline static vec4 xor(vec4 const& in0, vec4 const& in1) {
			return vec4(_mm_xor_ps(*in0.data, *in1.data));
		}
		__forceinline vec4 andnot(vec4 const& other) const {
			return vec4(_mm_andnot_ps(*data, *other.data));
		}
		
		__forceinline static vec4 dot(vec4 const& v0, vec4 const& v1) {
			vec4 mul = v0 * v1;
			vec4 tmp = _mm_shuffle_ps(*mul.data, *mul.data, _MM_SHUFFLE(3, 2, 1, 0));
			tmp += mul;
			vec4 res = _mm_shuffle_ps(*tmp.data, *tmp.data, _MM_SHUFFLE(2, 3, 0, 1));
			res += tmp;
			return res;
		}
};
}
#endif

#endif // !SIMD_SSE_H
