#pragma once
#ifndef SIMD_WRAP_VECTOR_HPP
#define SIMD_WRAP_VECTOR_HPP
#include "simd_traits.hpp"

namespace sw {

    template<typename T, size_t LEN>
    struct vector {
        using underlying_vector_type = typename simd_traits<T, LEN>::vector_type;
        static_assert(is_simd_compatible<T, LEN>, "Given combination of data type T and vector length LEN is not SIMD-compatible!");
        constexpr vector() noexcept;
        constexpr explicit vector(T value) noexcept;
        // not the same as LEN. Should we force matching 
        // for LEN or for size()?
        // - probably LEN, as thats the users/mathematical intent
        constexpr size_t size() const noexcept;
    private:
        underlying_vector_type data;
    };

}

#include "vector.inl"

#endif // !SIMD_WRAP_VECTOR_HPP
