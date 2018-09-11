#pragma once
#ifndef SIMD_WRAP_SIMD_TRAITS_HPP
#define SIMD_WRAP_SIMD_TRAITS_HPP
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <immintrin.h>

namespace sw {

    struct x64_platform_tag {};
    struct x86_platform_tag {};
    struct arm_platform_tag {};

    // will be set by CMake script eventually. will be used
    // to enable and use NEON types on ARM.
    using platform_type = x64_platform_tag;
    // will also be set by CMake, eventually
    static constexpr bool USE_AVX_INTRINSICS = true;

    namespace detail {

        template<typename T, size_t LEN>
        struct vector_type_proxy {
            constexpr static auto get_type() noexcept {
                if constexpr (!std::is_same_v<platform_type, arm_platform_tag>) {
                    if constexpr (std::is_same_v<T, float> && LEN <= 4) {
                        return __m128();
                    }
                    else if constexpr (std::is_same_v<T, double> && LEN <= 2) {
                        return __m128d();
                    }
                    else if constexpr (std::is_same_v<platform_type, x64_platform_tag> && USE_AVX_INTRINSICS) {
                        if constexpr (std::is_same_v<T, double> && LEN <= 4 && LEN > 2) {
                            return __m256d();
                        }
                        else if constexpr (std::is_same_v<T, float> && LEN <= 8 && LEN > 4) {
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
            using type = decltype(get_type());
        };

    }

    template<typename T, size_t LEN>
    constexpr bool is_simd_compatible = !std::is_same_v<T, detail::vector_type_proxy<T, LEN>::type >> ;

    template<typename T, size_t LEN>
    constexpr size_t vectorized_alignment = alignof(detail::vector_type_proxy<T, LEN>::type);

    template<typename T, size_t LEN>
    struct simd_traits {
        using vector_type = detail::vector_type_proxy<T, LEN>::type;
        constexpr static size_t num_entries = sizeof(vector_type) / sizeof(T);
        constexpr static size_t remainder_entries = num_entries % LEN;
        constexpr static size_t alignment = vectorized_alignment<T, LEN>;
    };

}
#endif //!SIMD_WRAP_SIMD_TRAITS_HPP
