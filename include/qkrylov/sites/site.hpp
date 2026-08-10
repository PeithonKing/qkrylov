#pragma once

#include "qkrylov/core/types.hpp"

#include "qkrylov/operators/local_action.hpp"

#include <string>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class Site
{
public:

    virtual ~Site() = default;

    virtual LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const = 0;
};

}
}
