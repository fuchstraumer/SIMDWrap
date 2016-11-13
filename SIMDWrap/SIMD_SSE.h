#pragma once
#ifndef SIMD_SSE_H
#define SIMD_SSE_H
#include "SIMD.h"
#ifdef SIMD_LEVEL_SSE3
#include <smmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
template<class T, std::size_t N>
class __declspec(align(N)) SIMDv {
public:
	SIMDv() {
		this->Data = static_cast<T*>(_aligned_malloc(sizeof(T), N));
	}
	~SIMDv() {
		_aligned_free(this->Data);
	}
	T* Data;
	T* alignedMalloc() {
		return static_cast<T*>(_aligned_malloc(sizeof(T), N));
	}
	void freeData() {
		_aligned_free(this->Data);
	}
};

namespace simd {
	class ivec4 : public SIMDv<__m128i, 16> {
	public:
		ivec4() : SIMDv() {
			*this->Data = _mm_setzero_si128();
		}
		ivec4(int32_t a, int32_t b, int32_t c, int32_t d = 0) {
			*this->Data = _mm_set_epi32(a, b, c, d);
		}
		ivec4(int32_t a) {
			*this->Data = _mm_set1_epi32(a);
		}
		ivec4(__m128i data) {
			*this->Data = data;
		}
		// Store the data from this vector into *data
		void Store(__m128i* data) {
			_mm_store_si128(data, *this->Data);
		}
		// Loads 128 bits of data into this->Data
		void Load(__m128i* data) {
			_mm_load_si128(data);
		}

		// Basic operations for building this vector
		// Set four separate elements a,b,c,d
		void Set(int32_t a = 0, int32_t b = 0, int32_t c = 0, int32_t d = 0) {
			*this->Data = _mm_set_epi32(a, b, c, d);
		}

		// Set all elements to be equal to x
		void Set(int32_t x) {
			*this->Data = _mm_set1_epi32(x);
		}

		// Get element at i and return a copy
		int32_t Get(uint8_t index) {
			return (this->Data->m128i_i32[index]);
		}

		// Basic operators
		__inline ivec4 ivec4::operator+(ivec4 const &other) {
			return ivec4(_mm_add_epi32(*this->Data, *other.Data));
		}
		__inline void operator+=(ivec4 const &other) const {
			*this->Data = _mm_add_epi32(*this->Data, *other.Data);
		}
		__inline ivec4 ivec4::operator-(ivec4 const &other) const {
			return ivec4(_mm_sub_epi32(*this->Data, *other.Data));
		}
		__inline void operator-=(ivec4 const &other) const {
			*this->Data = _mm_sub_epi32(*this->Data, *other.Data);
		}
		__inline ivec4 operator*(ivec4 const &other) const {
			return ivec4(_mm_mul_epi32(*this->Data, *other.Data));
		}
		__inline ivec4& operator*=(ivec4 const &other) const {
			*this->Data = _mm_mul_epi32(*other.Data, *this->Data);
		}
		__inline ivec4 operator<(ivec4 const &other) const {
			return ivec4(_mm_cmplt_epi32(*this->Data, *other.Data));
		}
		__inline ivec4 operator>(ivec4 const &other) const {
			return ivec4(_mm_cmpgt_epi32(*this->Data, *other.Data));
		}
		__inline ivec4 operator&(ivec4 const &other) const {
			return ivec4(_mm_and_si128(*this->Data, *other.Data));
		}
		__inline ivec4 operator|(ivec4 const &other) const {
			return ivec4(_mm_or_si128(*this->Data, *other.Data));
		}
		__inline static ivec4 xor(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_xor_si128(*in0.Data, *in1.Data));
		}
		__inline static ivec4 andnot(ivec4 const &in0, ivec4 const &in1) {
			return ivec4(_mm_andnot_si128(*in0.Data, *in0.Data));
		}
		__inline ivec4 operator~() const{
			ivec4 other(0xffffffff);
			return ivec4(_mm_xor_si128(*this->Data, *other.Data));
		}
		__inline ivec4 operator >> (ivec4 const& other) {

		}
		// More advanced mathematical functions.
		__inline ivec4 Max(ivec4& other) const {
			return ivec4(_mm_max_epi32(*this->Data, *other.Data));
		}
		__inline ivec4 Max(ivec4 const &in0, ivec4 const &in1) const {
			return ivec4(_mm_max_epi32(*in0.Data, *in1.Data));
		}
		// Convert to float type
		__inline static vec4 ConvertToFloat(ivec4 const &in) {
			return vec4(_mm_cvtepi32_ps(*in.Data));
		}
		__inline static ivec4 CastToInt(vec4 const &in) {
			return ivec4(_mm_castps_si128(*in.Data));
		}
	};

	class vec4 : public SIMDv<__m128, 16> {
	public:
		vec4() {
			*this->Data = _mm_setzero_ps();
		}
		vec4(float x, float y = 0.0f, float z = 0.0f, float w = 0.0f) {
			*this->Data = _mm_set_ps(x, y, z, w);
		}
		vec4(__m128 data) {
			*this->Data = data;
		}
		void SetOne(float a = 0.0f) {
			*this->Data = _mm_set_ps1(a);
		}
		// Basic operators
		// Unary operators
		__inline vec4& operator+=(vec4 const & other) const {
			*this->Data = _mm_add_ps(*this->Data, *other.Data);
		}
		__inline vec4& operator-=(vec4 const & other) const {
			*this->Data = _mm_add_ps(*this->Data, *other.Data);
		}
		__inline vec4& operator*=(vec4 const & other) const {
			*this->Data = _mm_mul_ps(*this->Data, *other.Data);
		}
		__inline vec4& operator/=(vec4 const & other) const {
			*this->Data = _mm_div_ps(*this->Data, *other.Data);
		}
		__inline vec4& operator++() const {
			vec4 increment; increment.SetOne(1.0f);
			*this->Data = _mm_add_ps(*this->Data, *increment.Data);
		}
		__inline vec4& operator--() const {
			vec4 decrement; decrement.SetOne(1.0f);
			*this->Data = _mm_add_ps(*this->Data, *decrement.Data);
		}
		// Binary operators
		__inline vec4 const operator+(vec4 const &other) const {
			return vec4(_mm_add_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator-(vec4 const &other) const {
			return vec4(_mm_sub_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator*(vec4 const &other) const {
			return vec4(_mm_mul_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator/(vec4 const &other) const {
			return vec4(_mm_div_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator<(vec4 const &other) const {
			return vec4(_mm_cmplt_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator>(vec4 const &other) const {
			return vec4(_mm_cmpgt_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator<=(vec4 const &other) const {
			return vec4(_mm_cmple_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator>=(vec4 const &other) const {
			return vec4(_mm_cmpge_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator&(vec4 const &other) const {
			return vec4(_mm_and_ps(*this->Data, *other.Data));
		}
		__inline vec4 const operator|(vec4 const& other) const {
			return vec4(_mm_or_ps(*this->Data, *other.Data));
		}
		__inline vec4 const xor(vec4 const& other) const {
			return vec4(_mm_xor_ps(*this->Data, *other.Data));
		}
		__inline vec4 const andnot(vec4 const& other) const {
			return vec4(_mm_andnot_ps(*this->Data, *other.Data));
		}

		// General mathematical functions
		__inline static vec4 const sqrt(vec4 const &in){
			return vec4(_mm_sqrt_ps(*in.Data));
		}
		__inline static vec4 const invSqrt(vec4 const &in){
			return vec4(_mm_rsqrt_ps(*in.Data));
		}
		__inline static vec4 const max(vec4 const &in0, vec4 const &in1){
			return vec4(_mm_max_ps(*in0.Data, *in1.Data));
		}
		__inline static vec4 const min(vec4 const &in0, vec4 const &in1){
			return vec4(_mm_min_ps(*in0.Data, *in1.Data));
		}
		__inline static vec4 const floor(vec4 const &in){
			return vec4(_mm_floor_ps(*in.Data));
		}
		__inline static ivec4 const convertToInt(vec4 const &in) {
			return ivec4(_mm_cvtps_epi32(*in.Data));
		}
		__inline static vec4 const CastToFloat(ivec4 const &in) {
			return vec4(_mm_castsi128_ps(*in.Data));
		}
	};
}
#endif

#endif // !SIMD_SSE_H
