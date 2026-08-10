#pragma once

#include "qkrylov/core/types.hpp"

#include "site.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class FermionSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied(
        StateID state,
        int site
    );

    static Real phase(
        StateID state,
        int site
    );
};

}

}
