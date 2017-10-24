#pragma once
#ifndef SIMD_SSE_H
#define SIMD_SSE_H
#include "SIMD.hpp"
#ifdef SIMD_LEVEL_SSE3
#include <cstdint>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>


namespace simd {

	class ivec4 {
	public:

		__m128i& SIMD_CALL data() noexcept {
			return _data;
		}

		const __m128i& SIMD_CALL data() const noexcept {
			return _data;
		}

		ivec4() {
			_data = _mm_setzero_si128();
		}

		explicit ivec4(const int32_t& a, const int32_t& b, const int32_t& c, const int32_t& d = 0) {
			_data = _mm_set_epi32(a, b, c, d);
		}

		explicit ivec4(const int32_t& a) {
			_data = _mm_set1_epi32(a);
		}

		explicit ivec4(__m128i _data) {
			_data = std::move(_data);
		}

		// Store the data from this vector into *data
		void SIMD_CALL Store(__m128i* result_data) {
			_mm_store_si128(result_data, _data);
		}
		// Loads 128 bits of data from *input_data
		void SIMD_CALL Load(__m128i* input_data) {
			_data = _mm_load_si128(input_data);
		}

		// Basic operations for building this vector
		// Set four separate elements a,b,c,d
		void SIMD_CALL Set(const int32_t& a = 0, const int32_t& b = 0, const int32_t& c = 0, const int32_t& d = 0) {
			_data = _mm_set_epi32(a, b, c, d);
		}

		// Set all elements to be equal to x
		void SIMD_CALL Set(const int32_t& x) {
			_data = _mm_set1_epi32(x);
		}

		// Basic operators
		// Unary operators
		ivec4& SIMD_CALL operator+=(ivec4 const &other) {
			_data = _mm_add_epi32(_data, other._data);
			return *this;
		}

		ivec4& SIMD_CALL operator-=(ivec4 const &other) {
			_data = _mm_sub_epi32(_data, other._data);
			return *this;
		}

		ivec4& SIMD_CALL operator*=(ivec4 const &other) {
			_data = _mm_mul_epi32(other._data, _data);
			return *this;
		}

		// Binary operators
		ivec4 SIMD_CALL ivec4::operator+(ivec4 const &other) const {
			return ivec4(_mm_add_epi32(_data, other._data));
		}

		ivec4 SIMD_CALL ivec4::operator-(ivec4 const &other) const {
			return ivec4(_mm_sub_epi32(_data, other._data));
		}

		ivec4 SIMD_CALL operator*(ivec4 const &other) const {
			return ivec4(_mm_mul_epi32(_data, other._data));
		}
		
		ivec4 SIMD_CALL operator<(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(other._data,_data));
		}

		ivec4 SIMD_CALL operator>(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(_data, other._data));
		}

		ivec4 SIMD_CALL operator&(ivec4 const &other) const {
			return ivec4(_mm_and_si128(_data, other._data));
		}

		ivec4 SIMD_CALL operator|(ivec4 const &other) const {
			return ivec4(_mm_or_si128(_data, other._data));
		}

		static ivec4 SIMD_CALL xor(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_xor_si128(in0._data, in1._data));
		}

		static ivec4 SIMD_CALL andnot(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_andnot_si128(in0._data, in0._data));
		}

		// performing a NOT on this vector is done by xor'ing with a vec
		// set to be 100% 1's
		ivec4 SIMD_CALL operator~() const{
			ivec4 other(0xffffffff);
			return ivec4(_mm_xor_si128(_data, other._data));
		}

		// Bit-shift this vector right using &dist as a mask
		// Fill using zeroes
		ivec4 SIMD_CALL operator>>(ivec4 const& dist) const {
			return ivec4(_mm_srl_epi32(_data, dist._data));
		}
		// Bit-shift this vector right by distance dist
		// Fill using zeroes
		ivec4 SIMD_CALL operator>>(int const& dist) const {
			ivec4 distv(dist);
			return ivec4(_mm_srl_epi32(_data, distv._data));
		}
		// Shift this vector left using &dist as a mask
		// Fill using zeroes
		ivec4 SIMD_CALL operator<<(ivec4 const& dist) const {
			return ivec4(_mm_sll_epi32(_data, dist._data));
		}
		// Shfit this vector left by distance dist
		// Fill using zeroes
		ivec4 SIMD_CALL operator<<(int const& dist) const {
			ivec4 distv(dist);
			return ivec4(_mm_sll_epi32(_data, distv._data));
		}

		ivec4 SIMD_CALL operator==(ivec4 const &other) const {
			return ivec4(_mm_cmpeq_epi32(_data,other._data));
		}

		// More advanced mathematical functions.
	private:
		__m128i _data;
	};

	class vec4 {
	public:

		__m128& SIMD_CALL data() noexcept {
			return _data;
		}

		const __m128& SIMD_CALL data() const noexcept {
			return _data;
		}

		// Initialize a vec4 with all elements set to zero
		vec4() {
			_data = _mm_setzero_ps();
		}

		// Initialize a vec4 with at least two distinct elements
		explicit vec4(const float& x, const float& y, const float& z = 0.0f, const float& w = 0.0f) {
			// Order in memory is actually reversed.
			_data = _mm_set_ps(w, z, y, x);
		}

		// Initialize a vec4 with all values set to a
		explicit vec4(const float& a) {
			_data = _mm_set1_ps(a);
		}

		// Initialize a vec4 by providing the intrinsic base type
		explicit vec4(__m128 _data) {
			_data = std::move(_data);
		}

		// Uniformly set all elements in this vector to a
		void SIMD_CALL Fill(const float& a = 0.0f) {
			_data = _mm_set_ps1(a);
		}

		// Load specified pointer into data
		void SIMD_CALL Load(float* ptr) {
			_data = _mm_load_ps(ptr);
		}

		// Basic operators

		// Unary operators
		const vec4& SIMD_CALL operator+=(vec4 const & other) noexcept {
			_data = _mm_add_ps(_data, other._data);
			return *this;
		}

		const vec4& SIMD_CALL operator-=(vec4 const & other) noexcept {
			_data = _mm_add_ps(_data, other._data);
			return *this;
		}

		const vec4& SIMD_CALL operator*=(vec4 const & other) noexcept {
			_data = _mm_mul_ps(_data, other._data);
			return *this;
		}

		const vec4& SIMD_CALL operator/=(vec4 const & other) noexcept {
			_data = _mm_div_ps(_data, other._data);
			return *this;
		}

		// pre & post increment/decrement operators

		const vec4& SIMD_CALL operator++() noexcept {
			static const __m128 one = _mm_set1_ps(1.0f);
			_data = _mm_add_ps(_data, one);
			return *this;
		}

		const vec4& SIMD_CALL operator--() noexcept {
			static const __m128 one = _mm_set1_ps(1.0f);
			_data = _mm_sub_ps(_data, one);
			return *this;
		}

		vec4 SIMD_CALL operator++(int) noexcept {
			static const __m128 one = _mm_set1_ps(1.0f);
			vec4 result = *this;
			_data = _mm_add_ps(_data, one);
			return result;
		}

		vec4 SIMD_CALL operator--(int) noexcept {
			static const __m128 one = _mm_set1_ps(1.0f);
			vec4 result = *this;
			_data = _mm_sub_ps(_data, one);
			return result;
		}

		// Binary operators
		vec4 SIMD_CALL operator+(const vec4& other) const noexcept {
			return vec4(_mm_add_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator-(const vec4& other) const noexcept {
			return vec4(_mm_sub_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator*(const vec4& other) const noexcept {
			return vec4(_mm_mul_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator/(const vec4& other) const noexcept {
			return vec4(_mm_div_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator<(const vec4& other) const noexcept {
			return vec4(_mm_cmplt_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator>(const vec4& other) const noexcept {
			return vec4(_mm_cmpgt_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator<=(const vec4& other) const noexcept {
			return vec4(_mm_cmple_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator>=(const vec4& other) const noexcept {
			return vec4(_mm_cmpge_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator&(const vec4& other) const noexcept {
			return vec4(_mm_and_ps(_data, other._data));
		}

		vec4 SIMD_CALL operator|(const vec4& other) const noexcept {
			return vec4(_mm_or_ps(_data, other._data));
		}

		// xor this vector with another vector
		vec4 SIMD_CALL xor(vec4 const& other) const {
			return vec4(_mm_xor_ps(_data, other._data));
		}
		
		// static functions (for operation on two distinct vectors)

		static vec4 SIMD_CALL xor(vec4 const& in0, vec4 const& in1) {
			return vec4(_mm_xor_ps(in0._data, in1._data));
		}
		vec4 SIMD_CALL andnot(vec4 const& other) const {
			return vec4(_mm_andnot_ps(_data, other._data));
		}
		
		static vec4 SIMD_CALL dot(vec4 const& v0, vec4 const& v1) {
			vec4 mul = v0 * v1;
			vec4 tmp(_mm_shuffle_ps(mul._data, mul._data, _MM_SHUFFLE(3, 2, 1, 0)));
			tmp += mul;
			vec4 res(_mm_shuffle_ps(tmp._data, tmp._data, _MM_SHUFFLE(2, 3, 0, 1)));
			res += tmp;
			return res;
		}

		static vec4 SIMD_CALL sqrt(const vec4& v) {
			return vec4(_mm_sqrt_ps(v.data()));
		}

	private:
		__m128 _data;
	};

	class mat4 {
	public:
	

	private:

		__m128 data[4];

	};
}
#endif

#endif // !SIMD_SSE_H
