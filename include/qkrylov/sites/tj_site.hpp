#pragma once

#include "qkrylov/core/types.hpp"

#include "site.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


class TJSite : public Site
{
public:

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

private:

    static bool occupied_up(
        StateID state,
        int site
    );

    static bool occupied_dn(
        StateID state,
        int site
    );

    static Real phase_up(
        StateID state,
        int site
    );

    static Real phase_dn(
        StateID state,
        int site
    );
};

}

}
