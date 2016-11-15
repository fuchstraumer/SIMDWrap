// SIMDWrap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SIMD.h"
#include "SIMD_Noise.h"
#include "SIMD_VecSet.h"
int main() {
	std::vector<simd::vec4> testparallel = simd::ParallelNoise(2048);
}

