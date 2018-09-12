#pragma once
#ifndef SIMD_WRAP_EXPRESSION_TEMPLATE_HELPERS_HPP
#define SIMD_WRAP_EXPRESSION_TEMPLATE_HELPERS_HPP
/*
    Following select how to refer to an expression template node,
    choosing by-value for scalars and by reference for the rest.
*/

namespace sw {

    namespace detail {

        template<typename T>
        class scalar;

        template<typename T>
        struct expr_node_traits {
            using reference_type = T const&;
        };

        template<typename T>
        struct expr_node_traits<scalar<T>> {
            using reference_type = scalar<T>;
        };

    }

}

#endif //!SIMD_WRAP_EXPRESSION_TEMPLATE_HELPERS_HPP
