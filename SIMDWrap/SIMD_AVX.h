#pragma once
#ifndef SIMD_AVX_H
#define SIMD_AVX_H

#include "SIMD.h"
#ifdef SIMD_LEVEL_AVX2
#include <immintrin.h>
#include <cstdlib>
namespace simd {
	class ivec8;
	class dvec4;

	class vec8 : public SIMDv<__m256, 32> {
	public:
		// Constructors
		vec8() : SIMDv() {
			this->Data = _mm256_setzero_ps();
		}

		vec8(__m256 data) {
			this->Data = data;
		}

		vec8(float a) {
			this->Data = _mm256_set1_ps(a);
		}

		vec8(float a, float b, float c = 0.0f, float d = 0.0f, float e = 0.0f, float f = 0.0f, float g = 0.0f, float h = 0.0f) {
			this->Data = _mm256_set_ps(h, g, f, e, d, c, b, a);
		}

		vec8(vec8 const &other) {
			this->Data = other.Data;
		}
		// Basic operators
		// Copy/equals
		__inline vec8& operator=(vec8 const &other) {
			this->Data = other.Data;
			return *this;
		}

		// Unary operators
		__inline vec8& operator+=(vec8 const &other) {
			this->Data = _mm256_add_ps(this->Data, other.Data);
			return *this;
		}
		__inline vec8& operator-=(vec8 const &other) {
			this->Data = _mm256_sub_ps(this->Data, other.Data);
			return *this;
		}
		__inline vec8& operator*=(vec8 const &other) {
			this->Data = _mm256_mul_ps(this->Data, other.Data);
			return *this;
		}
		__inline vec8& operator/=(vec8 const &other) {
			this->Data = _mm256_div_ps(this->Data, other.Data);
			return *this;
		}
		// Unary logic operators
		__inline vec8& operator&=(vec8 const &a) {
			this->Data = _mm256_and_ps(this->Data, a.Data);
			return *this;
		}
		__inline vec8& operator|=(vec8 const &a) {
			this->Data = _mm256_or_ps(this->Data, a.Data);
			return *this;
		}

		// Increment/Decrement operators
		__inline vec8& operator++() {
			vec8 one(1.0f);
			*this += one;
			return *this;
		}
		__inline vec8& operator--() {
			vec8 one(1.0f);
			*this -= one;
			return *this;
		}

		// Binary operators
		__inline vec8 operator+(vec8 const &other) const {
			return vec8(_mm256_add_ps(this->Data, other.Data));
		}
		__inline vec8 operator-(vec8 const &other) const {
			return vec8(_mm256_sub_ps(this->Data, other.Data));
		}
		__inline vec8 operator*(vec8 const &other) const {
			return vec8(_mm256_mul_ps(this->Data, other.Data));
		}
		__inline vec8 operator/(vec8 const &other) const {
			return vec8(_mm256_div_ps(this->Data, other.Data));
		}

		// Binary logic operators
		__inline vec8 operator&(vec8 const &other) const {
			return vec8(_mm256_and_ps(this->Data, other.Data));
		}
		__inline vec8 operator|(vec8 const &other) const {
			return vec8(_mm256_or_ps(this->Data, other.Data));
		}

		// Store

		// Load

		// Conversion - uses truncation
		__inline static ivec8 ConvertToInt(vec8 const& a) {
			return ivec8(_mm256_cvttps_epi32(a.Data));
		}

		// Casting - no overhead, converts directly.
		__inline static ivec8 CastToInt(vec8 const& a) {
			return ivec8(_mm256_castps_si256(a.Data));
		}

		// Constexpr version of previous, since castps is for compilation only
		__inline static constexpr __m256i CastToInt(__m256 const& a) {
			return _mm256_castps_si256(a);
		}
	};

	class ivec8 : public SIMDv<__m256i, 32> {
	public:
		// Constructors
		ivec8() : SIMDv() {
			this->Data = _mm256_setzero_si256();
		}

		ivec8(__m256i data) {
			this->Data = data;
		}

		ivec8(__m256 data) {
			this->Data = _mm256_cvttps_epi32(data);
		}

		ivec8(float a) {
			this->Data = _mm256_set1_epi32(a);
		}

		ivec8(float a, float b, float c = 0.0f, float d = 0.0f, float e = 0.0f, float f = 0.0f, float g = 0.0f, float h = 0.0f) {
			this->Data = _mm256_set_epi32(h, g, f, e, d, c, b, a);
		}

		ivec8(ivec8 const &other) {
			this->Data = other.Data;
		}

		// Conversion
		__inline vec8 ConvertToFloat(ivec8 const &a) {
			return vec8(_mm256_cvtepi32_ps(a.Data));
		}

		// Operators

		// Unary arithmetic operators
		__inline ivec8& operator+=(ivec8 const &other) {
			this->Data = _mm256_add_epi32(this->Data, other.Data);
			return *this;
		}
		__inline ivec8& operator-=(ivec8 const &other) {
			this->Data = _mm256_sub_epi32(this->Data, other.Data);
			return *this;
		}
		__inline ivec8 operator*=(ivec8 const &other) {
			this->Data = _mm256_mul_epi32(this->Data, other.Data);
			return *this;
		}
		// Warning: divide function requires conversions, may (will!) be inaccurate!
		__inline ivec8 operator/=(ivec8 const &other) {
			this->Data = vec8::ConvertToInt(_mm256_div_ps(ivec8::ConvertToFloat(*this).Data, ivec8::ConvertToFloat(other).Data)).Data;
			return *this;
		}

		// Unary logic operators
		__inline ivec8 const operator~() const {

		}
		__inline ivec8& operator&=(ivec8 const& a) {
			this->Data = _mm256_and_si256(this->Data, a.Data);
			return *this;
		}
		__inline ivec8& operator|=(ivec8 const& a) {
			this->Data = _mm256_or_si256(this->Data, a.Data);
			return *this;
		}

		// Increment and decrement operators
		__inline ivec8& operator++() {
			ivec8 one(1);
			this->Data = _mm256_add_epi32(this->Data, one.Data);
			return *this;
		}
		__inline ivec8& operator--() {
			ivec8 one(1);
			this->Data = _mm256_sub_epi32(this->Data, one.Data);
			return *this;
		}

		// Binary arithmetic operators
		__inline ivec8 const operator+(ivec8 const &a) const {
			return ivec8(_mm256_add_epi32(this->Data, a.Data));
		}
		__inline ivec8 const operator-(ivec8 const &a) const {
			return ivec8(_mm256_sub_epi32(this->Data, a.Data));
		}
		__inline ivec8 const operator*(ivec8 const &a) const {
			return ivec8(_mm256_mul_epi32(this->Data, a.Data));
		}

		// No division operand

		// Binary logic operators
		__inline ivec8 const operator&(ivec8 const& a) const {
			return ivec8(_mm256_and_si256(this->Data, a.Data));
		}
		__inline ivec8 const operator|(ivec8 const& a) const {
			return ivec8(_mm256_or_si256(this->Data, a.Data));
		}
	};

	class dvec4 : public SIMDv<__m256d, 32>{
	public:
		dvec4() {
			this->Data = _mm256_setzero_pd();
		}
		dvec4(double a) {
			this->Data = _mm256_set1_pd(a);
		}
		dvec4(double a, double b, double c, double d) {
			// Order in memory is effectively reversed.
			this->Data = _mm256_set_pd(d, c, b, a);
		}
	};
}
#endif // SIMD_LEVEL_AVX


#endif // !SIMD_AVX_H
