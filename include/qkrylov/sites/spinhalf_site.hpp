#pragma once

#include "qkrylov/core/types.hpp"

#include "site.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class SpinHalfSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool spin_up(
        StateID state,
        int site
    );
};

}
}
