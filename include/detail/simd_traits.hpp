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
    // will be set by CMake, eventually. this works for now: AVX is supported on 95% of hardware
    // with x64 according to steam hardware survey iirc
    static constexpr bool USE_AVX_INTRINSICS = !std::is_same_v<platform_type, x86_platform_tag> 
        && !std::is_same_v<platform_type, arm_platform_tag>;

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
                    else if constexpr (USE_AVX_INTRINSICS && std::is_floating_point_v<T>) {
                        if constexpr (std::is_same_v<T, double> && LEN <= 4 && LEN > 2) {
                            return __m256d();
                        }
                        else if constexpr (std::is_same_v<T, float> && LEN <= 8 && LEN > 4) {
                            return __m256();
                        }
                    }
                    else if constexpr (std::is_integral_v<T>) {// will have to cast from unsigned types, as these only accept signed
                    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                        if constexpr (LEN > 16 && USE_AVX_INTRINSICS) {
                            return __m256i();
                        }
                        else {
                            static_assert(LEN <= 16, "Length of integer vector is greater than supported on platform.");
                            return __m128i();
                        }
                        }
                        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { 
                            if constexpr (LEN > 8 && USE_AVX_INTRINSICS) {
                                return __m256i();
                            }
                            else {
                                static_assert(LEN <= 8, "Length of integer vector is greater than supported on platform.");
                                return __m128i();
                            }
                        }
                        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
                            if constexpr (LEN > 4 && USE_AVX_INTRINSICS) {
                                return __m128i();
                            }
                            else {
                                static_assert(LEN <= 4, "Length of integer vector is greater than supported on platform.");
                                return _mm_set1_epi32(static_cast<int>(val));
                            }
                        }
                        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
                            if constexpr (LEN > 2 && USE_AVX_INTRINSICS) {
                                return __m256i();
                            }
                            else {
                                static_assert(LEN <= 2, "Length of integer vector is greater than supported on platform.");
                                return __m128i();
                            }
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
        using value_type = T;
        constexpr static size_t num_entries = sizeof(vector_type) / sizeof(T);
        constexpr static size_t remainder_entries = num_entries % LEN;
        constexpr static size_t alignment = vectorized_alignment<T, LEN>;
    };

}
#endif //!SIMD_WRAP_SIMD_TRAITS_HPP
