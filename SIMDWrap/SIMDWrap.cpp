// SIMDWrap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "SIMD.h"

int main(){

	simd::SIMD4iv v1(16,32,64,128), v2(2), v3;
	
	v1 = v1 * v2;
	v2 += v3;
	v3 -= v2;
    return 0;
}

