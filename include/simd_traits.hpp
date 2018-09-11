#pragma once
#ifndef SIMD_WRAP_SIMD_TRAITS_HPP
#define SIMD_WRAP_SIMD_TRAITS_HPP
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <immintrin.h>

struct x64_platform_tag{};
struct x86_platform_tag{};
struct arm_platform_tag{};

// will be set by CMake script eventually
using platform_type = x64_platform_tag;
// will also be set by CMake, eventually
inline static constexpr bool USE_AVX_INTRINSICS = true;

namespace detail {

    template<typename T, size_t SIZE>
    constexpr auto get_vector_type() noexcept {
        if constexpr (!std::is_same_v<platform_type, arm_platform_tag>) {
            if constexpr (std::is_same_v<T, float> && SIZE <= 4) {
                return __m128();
            }
            else if constexpr (std::is_same_v<T, double> && SIZE <= 2) {
                return __m128d();
            }
            else if constexpr (std::is_same_v<platform_type, x64_platform_tag> && USE_AVX_INTRINSICS) { 
                if constexpr (std::is_same_v<T, double> && SIZE <= 4 && SIZE > 2) {
                    return __m256d();
                }
                else if constexpr (std::is_same_v<T, float> && SIZE <= 8 && SIZE > 4) {
                    return __m256();
                }
                else if constexpr (std::is_integral_v<T> && USE_AVX_INTRINSICS) {
                    return __m256i();
                }
            }
        }
        else {
            // need to add ARM case here.
            return T();
        }
    }

    template<typename T, size_t SIZE>
    using vectorized_type = decltype(get_vector_type<T,SIZE>());

    template<typename T, size_t SIZE>
    constexpr bool is_simd_compatible =!std::is_same_v<T, vectorized_type<T, SIZE>>;

    template<typename T, size_t SIZE>
    constexpr size_t vectorized_alignment = alignof(std::conditional_t<!std::is_same_v<void, vectorized_type<T, SIZE>>, vectorized_type<T,SIZE>, T>);

    template<typename T, size_t LEN, typename std::enable_if<is_simd_compatible<T,LEN>>::type* = nullptr>
    struct simd_traits {
        using vector_type = vectorized_type<T,LEN>;
        constexpr static size_t num_entries = sizeof(vector_type) / sizeof(data_type);
        constexpr static size_t remainder_entries = num_entries / LEN;
        constexpr static size_t alignment = vectorized_alignment<T,LEN>;
    };

}

#endif //!SIMD_WRAP_SIMD_TRAITS_HPP
