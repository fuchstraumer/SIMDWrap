#include "vector.hpp"

namespace sw {

    namespace detail {

        template<typename T, size_t LEN>
        auto broadcast_val(T val) noexcept {
            using vector_type = typename simd_traits<T, LEN>::vector_type;
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

            }
        }

    }

    template<typename T, size_t LEN>
    constexpr inline vector<T, LEN>::vector() noexcept {

    }

    template<typename T, size_t LEN>
    template<typename ...Args>
    inline constexpr vector<T, LEN>::vector(Args ...args) noexcept {
        if constexpr (sizeof...(Args) == 1) {
            data = detail::broadcast_val(args...);
        }
        else {

        }
    }

}
