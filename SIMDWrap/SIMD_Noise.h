#ifndef SIMD_NOISE_H
#define SIMD_NOISE_H
#include "SIMD_SSE.h"
#include "SIMD_Math.h"
#include "SIMD_Constants.h"
#include <random>
namespace simd {


// Following are noise functions for SSE3/4.1 intrinsics.

#ifdef SIMD_LEVEL_SSE3

	// "Ease curve" for noise values.
	static vec4 quintic_interp(vec4 const& input);
	
	static ivec4 hash(ivec4 const &seed, ivec4 const& x, ivec4 const& y, ivec4 const& z);

	static vec4 gradientcoord(ivec4 const &seed, ivec4 const &xi, ivec4 const &yi, ivec4 const &zi, vec4 const &x, vec4 const& y, vec4 const &z);
	
	static vec4 simplex(ivec4 const &seed, vec4 const &x, vec4 const &y, vec4 const &z);

	static float FBM(ivec4 const &seed, vec4 const &xi, vec4 const &yi, vec4 const &zi, float frequency, int octaves, float lacunarity, float gain);

	static vec4 RidgedMulti(ivec4 const &seed, vec4 const &xi, vec4 const &yi, vec4 const &zi, float frequency, int octaves, float lacunariy, float gain);

	static std::vector<vec4> ParallelNoise(unsigned int num) {
		std::vector<vec4> results; std::mt19937 r;
		std::uniform_int_distribution<> distr(-40, 40);
		results.resize(num);
#pragma loop(hint_parallel(4))
		for (unsigned int i = 0; i < num; ++i) {
			vec4 x = vec4((float)i + (float)distr(r), (float)i + (float)distr(r), (float)i + distr(r), (float)i + (float)distr(r));
			vec4 y = vec4((float)i + (float)distr(r), (float)i + (float)distr(r), (float)i + distr(r), (float)i + (float)distr(r));
			vec4 z = vec4((float)i + (float)distr(r), (float)i + (float)distr(r), (float)i + distr(r), (float)i + (float)distr(r));
			vec4 res = FBM(ivec4(2342, 42111, 362, 2132), x, y, z, 0.01f, 6, 2.5f, 1.0f);
			vec4 store; res.Store(store.Data);
			results[i] = (store);
		}
		return results;
	}

#endif // SIMD_LEVEL_SSE3


#ifdef SIMD_LEVEL_AVX2
	// Following are noise functions for AVX2 instructions.
	// Operate on vec8's

#endif // SIMD_LEVEL_AVX2

}


#endif // !SIMD_TERRAIN_H
