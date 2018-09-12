#pragma once
#ifndef SIMD_WRAP_EXPRESSION_TEMPLATES_BASE_EXPRESSIONS_HPP
#define SIMD_WRAP_EXPRESSION_TEMPLATES_BASE_EXPRESSIONS_HPP
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "expr_helpers.hpp"
#include "simd_traits.hpp"
namespace sw {

    namespace detail {

        template<typename T, size_t LEN>
        decltype(auto) broadcast_val(T val) noexcept {
            using vector_type = typename simd_traits<T, LEN>::vector_type;
            if constexpr (!std::is_same_v<platform_type, arm_platform_tag>) {
                if constexpr (std::is_same_v<vector_type, __m128>) {
                    return _mm_set1_ps(val);
                }
                else if constexpr (std::is_same_v<vector_type, __m128d>) {
                    return _mm_set1_pd(val);
                }
                else if constexpr (std::is_same_v<vector_type, __m256>) {
                    return _mm256_set1_ps(val);
                }
                else if constexpr (std::is_same_v<vector_type, __m256d>) {
                    return _mm256_set1_pd(val);
                }
                else if constexpr (std::is_integral_v<T>) {
                    // will have to cast from unsigned types, as these only accept signed
                    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                        if constexpr (LEN > 16 && USE_AVX_INTRINSICS) {
                            return _mm256_set1_epi8(static_cast<char>(val));
                        }
                        else {
                            static_assert(LEN <= 16, "Length of integer vector is greater than supported on platform.");
                            return _mm_set1_epi8(static_cast<char>(val));
                        }
                    }
                    else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { 
                        if constexpr (LEN > 8 && USE_AVX_INTRINSICS) {
                            return _mm256_set1_epi16(static_cast<short>(val));
                        }
                        else {
                            static_assert(LEN <= 8, "Length of integer vector is greater than supported on platform.");
                            return _mm_set1_epi16(static_cast<short>(val));
                        }
                    }
                    else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
                        if constexpr (LEN > 4 && USE_AVX_INTRINSICS) {
                            return _mm256_set1_epi32(static_cast<int>(val));
                        }
                        else {
                            static_assert(LEN <= 4, "Length of integer vector is greater than supported on platform.");
                            return _mm_set1_epi32(static_cast<int>(val));
                        }
                    }
                    else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
                        if constexpr (LEN > 2 && USE_AVX_INTRINSICS) {
                            return _mm256_set1_epi64x(static_cast<long long>(val));
                        }
                        else {
                            static_assert(LEN <= 2, "Length of integer vector is greater than supported on platform.");
                            return _mm_set1_epi64x(static_cast<long long>(val));
                        }
                    }
                }
            }
            else {
                // need to implement ARM intrinsics
            }
        }

    }

    /*
        Stores a scalar value, then when evaluated
        returns the appropriate vector type for the 
        current template parameters - this vector
        has all it's entries set to the scalar value
    */
    template<typename T, size_t LEN>
    struct broadcast_expression {
        static_assert(is_simd_compatible<T,LEN>, "Given template parameters cannot generate a valid vector type.");
        const T value;
    public:
        broadcast_expression(T val) noexcept : value(std::move(val)) {}
        typename simd_traits<T,LEN>::vector_type operator()() const noexcept {
            return detail::broadcast_val<T,LEN>(value);
        }
    };

}

#endif //!SIMD_WRAP_EXPRESSION_TEMPLATES_BASE_EXPRESSIONS_HPP
