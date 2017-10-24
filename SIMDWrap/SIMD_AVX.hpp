#pragma once
#ifndef SIMD_AVX_H
#define SIMD_AVX_H

#include "SIMD.hpp"
#ifdef SIMD_LEVEL_AVX2
#include <immintrin.h>
#include <cstdlib>
namespace simd {
	class ivec8;
	class dvec4;

	class vec8 {
	public:

		__m256& SIMD_CALL data() noexcept {
			return _data;
		}

		const __m256& SIMD_CALL data() const noexcept {
			return _data;
		}

		vec8()  {
			_data= _mm256_setzero_ps();
		}

		explicit vec8(__m256 _data) {
			_data = std::move(_data);
		}

		explicit vec8(const float& a)  {
			_data = _mm256_set1_ps(a);
		}

		explicit vec8(const float& a, const float& b, const float& c = 0.0f, const float& d = 0.0f, const float& e = 0.0f, const float& f = 0.0f, const float& g = 0.0f, const float& h = 0.0f) {
			_data = _mm256_set_ps(h, g, f, e, d, c, b, a);
		}

		// Unary operators
		vec8& SIMD_CALL operator+=(vec8 const &other) {
			_data = _mm256_add_ps(_data, other._data);
			return *this;
		}
		vec8& SIMD_CALL operator-=(vec8 const &other) {
			_data = _mm256_sub_ps(_data, other._data);
			return *this;
		}
		vec8& SIMD_CALL operator*=(vec8 const &other) {
			_data = _mm256_mul_ps(_data, other._data);
			return *this;
		}
		vec8& SIMD_CALL operator/=(vec8 const &other) {
			_data = _mm256_div_ps(_data, other._data);
			return *this;
		}

		// Unary logic operators
		vec8& SIMD_CALL operator&=(vec8 const &a) {
			_data = _mm256_and_ps(_data, a._data);
			return *this;
		}
		vec8& SIMD_CALL operator|=(vec8 const &a) {
			_data = _mm256_or_ps(_data, a._data);
			return *this;
		}

		// Increment/Decrement operators

		const vec8 SIMD_CALL operator++(int) {
			__m256 incr = _mm256_set1_ps(1.0f);
			vec8 res = *this;
			_data = _mm256_add_ps(_data, incr);
			return res;
		}

		const vec8& SIMD_CALL operator++() {
			__m256 incr = _mm256_set1_ps(1.0f);
			_data = _mm256_add_ps(_data, incr);
			return *this;
		}

		const vec8 SIMD_CALL operator--(int) {
			__m256 incr = _mm256_set1_ps(1.0f);
			vec8 res = *this;
			_data = _mm256_sub_ps(_data, incr);
			return res;
		}

		vec8& SIMD_CALL operator--() {
			__m256 incr = _mm256_set1_ps(1.0f);
			_data = _mm256_sub_ps(_data, incr);
			return *this;
		}

		// Binary operators
		vec8 SIMD_CALL operator+(vec8 const &other) const noexcept {
			return vec8(_mm256_add_ps(_data, other._data));
		}
		vec8 SIMD_CALL operator-(vec8 const &other) const noexcept {
			return vec8(_mm256_sub_ps(_data, other._data));
		}
		vec8 SIMD_CALL operator*(vec8 const &other) const {
			return vec8(_mm256_mul_ps(_data, other._data));
		}
		vec8 SIMD_CALL operator/(vec8 const &other) const {
			return vec8(_mm256_div_ps(_data, other._data));
		}

		// Binary logic operators
		vec8 SIMD_CALL operator&(vec8 const &other) const {
			return vec8(_mm256_and_ps(_data, other._data));
		}
		vec8 SIMD_CALL operator|(vec8 const &other) const {
			return vec8(_mm256_or_ps(_data, other._data));
		}

		// Binary comparison operators
		vec8 SIMD_CALL operator>(vec8 const &other) const {
			return vec8(_mm256_cmp_ps(_data, other._data, _CMP_GT_OQ));
		}
		vec8 SIMD_CALL operator>=(vec8 const &other) const {
			return vec8(_mm256_cmp_ps(_data, other._data, _CMP_GE_OQ));
		}
		vec8 SIMD_CALL operator<(vec8 const &other) const {
			return vec8(_mm256_cmp_ps(_data, other._data, _CMP_LT_OQ));
		}
		vec8 SIMD_CALL operator<=(vec8 const &other) const {
			return vec8(_mm256_cmp_ps(_data, other._data, _CMP_LE_OQ));
		}

		// Math operators
		vec8 SIMD_CALL floor(const vec8 &in) const {
			return vec8(_mm256_floor_ps(in._data));
		}

		// Special comparison operators
		static vec8 SIMD_CALL and_not(const vec8& v0, const vec8& v1) {
			return vec8(_mm256_andnot_ps(v0._data, v1._data));
		}

		static vec8 SIMD_CALL xor(const vec8& v0, const vec8& v1) {
			return vec8(_mm256_xor_ps(v0._data, v1._data));
		}

		/*
			Following defined in SIMD_Math.h
		*/

		// Takes the max of the input vector and returns a vector
		// where each entry is the maximum of the two at a position
		static vec8 SIMD_CALL max(vec8 const & a, vec8 const & b);

		// Takes the min of the input vector and returns a vector
		// where each entry is the minimum of the two at a position
		static vec8 SIMD_CALL min(vec8 const & a, vec8 const & b);

		// Takes the sqrt of the input vector
		static vec8 SIMD_CALL sqrt(vec8 const & a);

	private:
		__m256 _data;
	};

	class ivec8 {
	private:
		__m256i _data;
	public:

		__m256i& SIMD_CALL data() noexcept {
			return _data;
		}

		const __m256i& SIMD_CALL data() const noexcept {
			return _data;
		}

		// Constructors
		ivec8() {
			_data = _mm256_setzero_si256();
		}

		ivec8(__m256i _data)  {
			_data = std::move(_data);
		}

		ivec8(const int32_t& a)  {
			_data = _mm256_set1_epi32(a);
		}

		ivec8(const int32_t& a, const int32_t& b, const int32_t& c = 0.0f, const int32_t& d = 0.0f, const int32_t& e = 0.0f, const int32_t& f = 0.0f, const int32_t& g = 0.0f, const int32_t& h = 0.0f) {
			_data = _mm256_set_epi32(h, g, f, e, d, c, b, a);
		}

		ivec8(ivec8 const &other){
			_data = other._data;
		}

		// Conversion
		static vec8 SIMD_CALL ConvertToFloat(ivec8 const &a) {
			return vec8(_mm256_cvtepi32_ps(a._data));
		}
		static vec8 SIMD_CALL CastToFloat(ivec8 const& a) {
			return vec8(_mm256_castsi256_ps(a._data));
		}

		// Operators

		// Unary arithmetic operators
		const ivec8& SIMD_CALL operator+=(ivec8 const &other) {
			_data = _mm256_add_epi32(_data, other._data);
			return *this;
		}

		const ivec8& SIMD_CALL operator-=(ivec8 const &other) {
			_data = _mm256_sub_epi32(_data, other._data);
			return *this;
		}

		const ivec8& SIMD_CALL operator*=(ivec8 const &other) {
			_data = _mm256_mul_epi32(_data, other._data);
			return *this;
		}

		// Unary logic operators
		const ivec8& SIMD_CALL operator&=(ivec8 const& a) {
			_data = _mm256_and_si256(_data, a._data);
			return *this;
		}

		const ivec8& SIMD_CALL operator|=(ivec8 const& a) {
			_data = _mm256_or_si256(_data, a._data);
			return *this;
		}

		// Increment and decrement operators
		const ivec8& SIMD_CALL operator++() {
			static const __m256i one = _mm256_set1_epi32(1);
			_data = _mm256_add_epi32(_data, one);
			return *this;
		}

		const ivec8 SIMD_CALL operator++(int) {
			static const __m256i one = _mm256_set1_epi32(1);
			ivec8 result = *this;
			_data = _mm256_add_epi32(_data, one);
			return result;
		}

		const ivec8& SIMD_CALL operator--() {
			static const __m256i one = _mm256_set1_epi32(1);
			_data = _mm256_sub_epi32(_data, one);
			return *this;
		}

		const ivec8 SIMD_CALL operator--(int) {
			static const __m256i one = _mm256_set1_epi32(1);
			ivec8 result = *this;
			_data = _mm256_sub_epi32(_data, one);
			return result;
		}

		ivec8 const SIMD_CALL operator+(ivec8 const &a) const {
			return ivec8(_mm256_add_epi32(_data, a._data));
		}
		ivec8 const SIMD_CALL operator-(ivec8 const &a) const {
			return ivec8(_mm256_sub_epi32(_data, a._data));
		}
		ivec8 const SIMD_CALL operator*(ivec8 const &a) const {
			return ivec8(_mm256_mul_epi32(_data, a._data));
		}

		ivec8 SIMD_CALL operator&(ivec8 const& a) const {
			return ivec8(_mm256_and_si256(_data, a._data));
		}
		ivec8 SIMD_CALL operator|(ivec8 const& a) const {
			return ivec8(_mm256_or_si256(_data, a._data));
		}

		ivec8 SIMD_CALL operator>>(int const& count) const {
			return ivec8(_mm256_slli_epi32(_data, count));
		}

		ivec8 SIMD_CALL operator<<(int const& count) const {
			return ivec8(_mm256_srli_epi32(_data, count));
		}

		ivec8 SIMD_CALL operator>(ivec8 const& other) const {
			return ivec8(_mm256_cmpgt_epi32(_data, other._data));
		}
		ivec8 SIMD_CALL operator<(ivec8 const& other) const {
			return ivec8(_mm256_cmpgt_epi32(other._data, _data));
		}
		ivec8 SIMD_CALL operator==(ivec8 const& other) const {
			return ivec8(_mm256_cmpeq_epi32(_data, other._data));
		}

		// Other operators/functions

		static ivec8 SIMD_CALL and_not(ivec8 const& v0, ivec8 const& v1) {
			return ivec8(_mm256_andnot_si256(v0._data, v1._data));
		}
		static ivec8 SIMD_CALL xor(ivec8 const& v0, ivec8 const& v1) {
			return ivec8(_mm256_xor_si256(v0._data, v1._data));
		}
		static ivec8 SIMD_CALL not(ivec8 const& v0) {
			return ivec8(_mm256_xor_si256(v0._data, ivec8(0xffffffff)._data));
		}

		// Dot product - taken from http://tomjbward.co.uk/simd-optimized-dot-and-cross
		static vec8 SIMD_CALL dot(const vec8& v0, const vec8& v1) {
			__m256 res = _mm256_mul_ps(v0.data(), v1.data());
			res = _mm256_hadd_ps(res, res);
			res = _mm256_hadd_ps(res, res);
			return vec8(res);
		}

		// Cross product, taken from http://tomjbward.co.uk/simd-optimized-dot-and-cross


	};

}
#endif // SIMD_LEVEL_AVX


#endif // !SIMD_AVX_H
