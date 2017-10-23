#pragma once
#ifndef SIMD_WRAP_SIMD_V_H
#define SIMD_WRAP_SIMD_V_H

#include <utility>
#include <memory>

#ifdef _MSC_VER
#define SIMD_CALL __vectorcall
#else 
#define SIMD_CALL __cdecl
#endif

// Set correct aligned memory allocation function based on OS/Compiler
#if defined(_WIN32)

template <typename T>
T* aligned_malloc(std::size_t alignment) {
	return static_cast<T*>(_aligned_malloc(sizeof(T), alignment));
};

template<typename T>
void aligned_free(void* data) {
	_aligned_free(data);
}

#elif defined(unix)

template<typename T>
T* aligned_malloc(std::size_t alignment) {
	return posix_memalign(sizeof(T), alignment);
}

template<typename T>
void aligned_free(void* data) {
	free(data);
}


#endif

template<class T, size_t N>
class simd_vector_t {
public:

	simd_vector_t() : data(aligned_malloc<T>(N), &aligned_free) {}

	~simd_vector_t() = default;

	simd_vector_t(const simd_vector_t& other) : data(aligned_malloc<T>(N), &aligned_free) {
		*data = *other.data;
	}

	simd_vector_t& operator=(const simd_vector_t& other) {
		*data = *other.data;
		return *this;
	}

	simd_vector_t(simd_vector_t&& other) noexcept : data(other.data) {
		other.data = nullptr;
	}

	simd_vector_t& operator=(simd_vector_t&& other) noexcept {
		data = other.data;
		other.data = nullptr;
		return *this;
	}

	T* alignedMalloc() {
		return aligned_malloc<T>(N);
	}

protected:
	std::unique_ptr<T> data;

};

#endif // !SIMD_WRAP_SIMD_V_H
