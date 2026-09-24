#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/sites/site.hpp"

#include <string>

namespace qkrylov {


class SpinSSite : public Site
{
public:

    int bits_per_site() const override { return -d_; }


    explicit SpinSSite(double S = 0.5);

    LocalAction apply(
        const std::string& op,
        int site,
        StateID state
    ) const override;

    double spin() const noexcept { return S_; }
    int dimension_per_site() const noexcept { return d_; }

private:

    double S_;
    int d_;
};


} // namespace qkrylov
