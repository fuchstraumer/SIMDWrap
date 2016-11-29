// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once


#include <stdio.h>
#include <tchar.h>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <iostream>

/*template<class T, size_t N>
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
};*/

template<class T, size_t N>
class __declspec(align(N)) SIMDv {
public:
	T Data;
};


// TODO: reference additional headers your program requires here
