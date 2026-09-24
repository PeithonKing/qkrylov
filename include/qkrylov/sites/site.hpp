#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/operators/local_action.hpp"
#include "qkrylov/core/instruction.hpp"

#include <string>
#include <vector>
#include <stdexcept>

namespace qkrylov {

class OperatorTerm; // Forward declaration

class Site
{
public:
    virtual int bits_per_site() const = 0;

    virtual ~Site() = default;

    virtual LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const = 0;

    // Compiles a parsed operator term into highly optimized VM instructions.
    virtual std::vector<Instruction> compile(const OperatorTerm& term) const {
        throw std::runtime_error("VM Compilation not implemented for this Site type.");
    }
};

} // namespace qkrylov
