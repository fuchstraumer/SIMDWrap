#pragma once
#ifndef SIMD_MATRICES_H
#define SIMD_MATRICES_H

#include "SIMD.hpp"
#include <array>
#ifdef SIMD_LEVEL_SSE3

// 4x4 matrix type using SIMD vectors
namespace simd {
	class mat4 {
	public:
		mat4(vec4& x, vec4& y, vec4 &z, vec4 &w) {

		}
	private:
		// Contains the actual data for this matrix
		std::array<vec4, 4> data;
	};
}
#endif // !SIMD_LEVEL_SSE3

#ifdef SIMD_LEVEL_AVX2
#include "SIMD_AVX.hpp"
namespace simd {

	// This matrix holds 3 8-element vectors, for use in 3D operations
	class mat8x3 {
	public:
		mat8x3() { }

		mat8x3(const vec8 &x, const vec8 &y, const vec8 &z) {
			data[0] = x;
			data[1] = y;
			data[2] = z;
		}

		mat8x3(const vec8& v) {
			data[0] = v;
			data[1] = v;
			data[2] = v;
		}


	private:
		std::array<vec8, 3> data;
	};

	class mat8x4 {
	public:
		mat8x4() = default;
		~mat8x4() = default;
	};
}

#endif // SIMD_LEVEL_AVX2


#endif // !SIMD_MATRICES_H
