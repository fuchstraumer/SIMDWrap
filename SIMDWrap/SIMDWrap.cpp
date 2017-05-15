// SIMDWrap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SIMD.h"


int main() {

	using namespace simd;

	vec8 vec0(0.0f);
	vec8 vec1(1.0f);
	vec0 = std::move(vec1);
	std::cerr << "hi" << std::endl;
	vec8 vec2(5.0f);
	vec2 += vec1;
	vec2++;
	vec1--;

	return 0;
}

