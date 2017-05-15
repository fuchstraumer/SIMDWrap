#pragma once
#ifndef SIMD_WRAP_SIMD_V_H
#define SIMD_WRAP_SIMD_V_H

// Set correct aligned memory allocation function based on OS/Compiler
#if defined(_WIN32)
template <typename T>
__forceinline T* aligned_malloc(std::size_t alignment) {
	return static_cast<T*>(_aligned_malloc(sizeof(T), alignment));
};

template<typename T>
__forceinline void aligned_free(void* data) {
	_aligned_free(data);
}
#endif

// Untested, but these "should" be correct.
#if defined(unix)
template<typename T>
__forceinline T* aligned_malloc(std::size_t alignment) {
	return posix_memalign(sizeof(T), alignment);
}

template<typename T>
__forceinline void aligned_free(void* data) {
	free(data);
}
#endif

template<class T, size_t N>
class __declspec(align(N)) SIMDv {
public:
	SIMDv() {
		data = aligned_malloc<T>(N);
	}

	~SIMDv() {
		aligned_free<T>(data);
	}
	// Custom destructor removed: seems compiler handles this better than I could.
	T* data;
	T* alignedMalloc() {
		return aligned_malloc<T>(N);
	}
	void free() {
		aligned_free(data);
	}
};

#endif // !SIMD_WRAP_SIMD_V_H
