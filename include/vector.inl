#include "vector.hpp"

namespace sw {

    template<typename T, size_t LEN>
    constexpr inline vector<T, LEN>::vector() noexcept {

    }

    template<typename T, size_t LEN>
    template<typename...Args>
    inline constexpr vector<T, LEN>::vector(Args...args) noexcept {
        if constexpr (sizeof...(Args) == 1) {
            data = detail::broadcast_val(args...);
        }
        else {
            static_assert(sizeof...(Args) == LEN, "Variadic constructor must have values for full length of vector, or single \"fill\" argument!");
        }
    }

}
