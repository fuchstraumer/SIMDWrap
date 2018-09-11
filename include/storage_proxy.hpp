#pragma once
#ifndef SIMD_WRAP_STORAGE_PROXY_HPP
#define SIMD_WRAP_STORAGE_PROXY_HPP
#include <type_traits>
#include <emmintrin.h>

template<typename T, size_t LEN>
struct storage_proxy;

inline static constexpr bool USE_AVX_INTRINSICS = true;

namespace detail {

    template<typename T, size_t SIZE>
    constexpr auto get_vector_type() noexcept {
        if constexpr (std::is_same_v<T, double> && SIZE <= 4 && SIZE > 2) {
            return __m256d();
        }
        else if constexpr (std::is_same_v<T, double> && SIZE <= 2) {
            return __m128d();
        }
        else if constexpr (std::is_same_v<T, float> && SIZE <= 8 && SIZE > 4) {
            return __m256();
        }
        else if constexpr (std::is_same_v<T, float> && SIZE <= 4) {
            return __m128();
        }
        else if constexpr (std::is_integral_v<T> && USE_AVX_INTRINSICS) {
            return __m256i();
        }
        else if constexpr (std::is_integral_v<T>) {
            return __m128i();
        }
        else {
            return T();
        }
    }

    template<typename T, size_t SIZE>
    using vectorized_type = decltype(get_vector_type<T,SIZE>());

    template<typename T, size_t SIZE>
    constexpr bool is_simd_compatible =!std::is_same_v<void, vectorized_type<T, SIZE>>;

    template<typename T, size_t SIZE>
    constexpr size_t vectorized_alignment = alignof(std::conditional_t<!std::is_same_v<void, vectorized_type<T, SIZE>>, vectorized_type<T,SIZE>, T>);

    template<typename T, size_t SIZE>
    void broadcast_simd(T val) {
        if constexpr (std::is_same_v<vectorized_type<T, SIZE>, __m256d>) {

        }
        else if constexpr (std::is_same_v<vectorized_type<T, SIZE>, __m128d>) {
            
        }
    }

    template<typename T, size_t LEN>
    struct simd_traits {

    };

}

#endif //!SIMD_WRAP_STORAGE_PROXY_HPP
