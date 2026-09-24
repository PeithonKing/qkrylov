#pragma once

#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/operators/local_op.hpp"

namespace qkrylov {
namespace models {

class Heisenberg1D : public OpSum
{
public:
    Heisenberg1D(int L, double J = 1.0, double Jz = 1.0, bool pbc = false)
    {
        for (int i = 0; i < (pbc ? L : L - 1); ++i) {
            int j = (i + 1) % L;
            add_term( (J / 2.0) * Sp(i) * Sm(j) );
            add_term( (J / 2.0) * Sm(i) * Sp(j) );
            add_term( Jz * Sz(i) * Sz(j) );
        }
    }
};

class TransverseFieldIsing1D : public OpSum
{
public:
    TransverseFieldIsing1D(int L, double J = 1.0, double h = 1.0, bool pbc = false)
    {
        for (int i = 0; i < (pbc ? L : L - 1); ++i) {
            int j = (i + 1) % L;
            add_term( -J * Sz(i) * Sz(j) );
        }
        for (int i = 0; i < L; ++i) {
            add_term( -h * Sx(i) );
        }
    }
};

} // namespace models
} // namespace qkrylov
