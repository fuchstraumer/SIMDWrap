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
template<class T, size_t N>
class __declspec(align(N)) SIMDv {
public:
	SIMDv() {
		this->Data = aligned_malloc<T>(N);
	}
	// Custom destructor removed: seems compiler handles this better than I could.
	T* Data;
	T* alignedMalloc() {
		return aligned_malloc<T>(N);
	}
	void freeData() {
		aligned_free(this->Data);
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
			*this->Data = _mm_load_si128(data);
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
			return ivec4(_mm_cmpgt_epi32(*other.Data,*this->Data));
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
		__inline ivec4 operator >> (ivec4 const& dist) {
			return ivec4(_mm_srl_epi32(*this->Data, *dist.Data));
		}
		__inline ivec4 operator >> (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_srl_epi32(*this->Data, *distv.Data));
		}
		__inline ivec4 operator << (ivec4 const& dist) {
			return ivec4(_mm_sll_epi32(*this->Data, *dist.Data));
		}
		__inline ivec4 operator << (int const& dist) {
			ivec4 distv(dist);
			return ivec4(_mm_sll_epi32(*this->Data, *distv.Data));
		}
		__inline ivec4 operator==(ivec4 const &other) const {
			return ivec4(_mm_cmpeq_epi32(*this->Data,*other.Data));
		}
		// More advanced mathematical functions.
		
	};

	class vec4 : public SIMDv<__m128, 16> {
	public:
		vec4() {
			*this->Data = _mm_setzero_ps();
		}
		vec4(float x, float y = 0.0f, float z = 0.0f, float w = 0.0f) {
			// Order in memory is actually reversed.
			*this->Data = _mm_set_ps(w, z, y, x);
		}
		vec4(__m128 data) {
			*this->Data = data;
		}
		void SetOne(float a = 0.0f) {
			*this->Data = _mm_set_ps1(a);
		}
		// Load/store operations
		void Store(__m128* data) {
			for (int i = 0; i < 4; ++i) {
				_mm_store_ps(&data->m128_f32[i], *this->Data);
			}
		}
		// Loads 128 bits of data into this->Data
		void Load(__m128* data) {
			__m128 v;
			v = _mm_load_ps(&data->m128_f32[0]);
			*this->Data = v;
		}
		// Basic operators
		// Unary operators
		__inline vec4& operator+=(vec4 const & other) {
			*this->Data = _mm_add_ps(*this->Data, *other.Data);
			return *this;
		}
		__inline vec4& operator-=(vec4 const & other) {
			*this->Data = _mm_add_ps(*this->Data, *other.Data);
			return *this;
		}
		__inline vec4& operator*=(vec4 const & other) {
			*this->Data = _mm_mul_ps(*this->Data, *other.Data);
			return *this;
		}
		__inline vec4& operator/=(vec4 const & other) {
			*this->Data = _mm_div_ps(*this->Data, *other.Data);
			return *this;
		}
		__inline vec4& operator++() {
			vec4 increment; increment.SetOne(1.0f);
			*this->Data = _mm_add_ps(*this->Data, *increment.Data);
			return *this;
		}
		__inline vec4& operator--() {
			vec4 decrement; decrement.SetOne(1.0f);
			*this->Data = _mm_add_ps(*this->Data, *decrement.Data);
			return *this;
		}
		// Binary operators
		__inline vec4 operator+(vec4 const &other) const {
			return vec4(_mm_add_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator-(vec4 const &other) const {
			return vec4(_mm_sub_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator*(vec4 const &other) const {
			return vec4(_mm_mul_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator/(vec4 const &other) const {
			return vec4(_mm_div_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator<(vec4 const &other) const {
			return vec4(_mm_cmplt_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator>(vec4 const &other) const {
			return vec4(_mm_cmpgt_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator<=(vec4 const &other) const {
			return vec4(_mm_cmple_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator>=(vec4 const &other) const {
			return vec4(_mm_cmpge_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator&(vec4 const &other) const {
			return vec4(_mm_and_ps(*this->Data, *other.Data));
		}
		__inline vec4 operator|(vec4 const& other) const {
			return vec4(_mm_or_ps(*this->Data, *other.Data));
		}
		__inline vec4 xor(vec4 const& other) const {
			return vec4(_mm_xor_ps(*this->Data, *other.Data));
		}
		__inline static vec4 xor(vec4 const& in0, vec4 const& in1) {
			return vec4(_mm_xor_ps(*in0.Data, *in1.Data));
		}
		__inline vec4 andnot(vec4 const& other) const {
			return vec4(_mm_andnot_ps(*this->Data, *other.Data));
		}
		
	};
}
#endif

#endif // !SIMD_SSE_H
