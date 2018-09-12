#pragma once
#ifndef SIMD_WRAP_EXPRESSION_TEMPLATE_BINARY_OPERATORS_HPP
#define SIMD_WRAP_EXPRESSION_TEMPLATE_BINARY_OPERATORS_HPP
#include "base_expressions.hpp"

namespace sw {

    namespace detail {

    }

    template<typename OP0, typename OP1>
    struct expression_add {

        using operand_0_ref_type = typename detail::expr_node_traits<OP0>::reference_type;
        using operand_1_ref_type = typename detail::expr_node_traits<OP1>::reference_type;

        expression_add(OP0& operand_0, OP1& operand_1) noexcept : operand0(operand_0), 
            operand1(operand_1) {}
        decltype(auto) operator()() noexcept {
            if constexpr (std::is_arithmetic_v<OP0> || std::is_arithmetic_v<OP1>) {
                // need to add broadcast operator
            }
        }
    };

}

#endif //!SIMD_WRAP_EXPRESSION_TEMPLATE_BINARY_OPERATORS_HPP
