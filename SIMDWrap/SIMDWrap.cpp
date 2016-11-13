// SIMDWrap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SIMD.h"
#include "SIMD_Noise.h"
#include "SIMD_VecSet.h"
int main() {
	simd::vec4 x(0.0f, 1.0f, 2.0f, 3.0f); simd::vec4 z; z = x;
	simd::vec4 y(0.0f, 1.0f, 2.0f, 3.0f); simd::ivec4 seed(21854, 2716, 63123, 12711);
	auto vecs = simd::vec4Set(0.0f, 64.0f, 0.0f, 128.0f, 0.0f, 64.0f, 0.1f);
	std::vector<float> results;
	for (int i = 0; i < vecs[0].size(); ++i) {
		float noiseresult = FBM(seed, vecs[0][i], vecs[1][i], vecs[2][i], 0.1f, 6, 1.5f, 1.0f);
		results.push_back(noiseresult);
	}
	return 0;
}

