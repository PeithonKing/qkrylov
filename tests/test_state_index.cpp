// StateIndex: mode selection, exact index for every basis state, rejection of non-members.
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/hamiltonian/state_index.hpp"
#include "qkrylov/core/device.hpp"

#include <cassert>
#include <iostream>
#include <set>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;
using Space = Kokkos::DefaultHostExecutionSpace;
using Idx = StateIndex<Space>;

static Idx build(const Basis& b) {
    std::vector<StateID> s(b.size());
    for (Index i = 0; i < b.size(); ++i) s[i] = b.state(i);
    Kokkos::View<StateID*, Space> d("states", s.size());
    for (Index i = 0; i < s.size(); ++i) d(i) = s[i];
    return build_state_index<Space>(s, b.conserved_masks(), d);
}

static void check(const char* name, const Basis& b, int expected_mode, int nbits, bool expect_complete = true) {
    const Idx idx = build(b);
    std::cout << name << ": dim=" << b.size() << " mode=" << idx.mode << " complete=" << idx.complete << "\n";
    assert(idx.mode == expected_mode);
    if (expected_mode == Idx::Lin) assert(idx.complete == expect_complete);
    std::set<StateID> members;
    for (Index i = 0; i < b.size(); ++i) {
        Index j = 0;
        assert(idx.find(b.state(i), j) && j == i);
        members.insert(b.state(i));
    }
    // Every other state in range (and a few beyond) must be rejected.
    for (StateID s = 0; s < (StateID(1) << nbits) + 4; ++s) {
        Index j = 0;
        if (!members.count(s)) assert(!idx.find(s, j));
    }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        { Sector s; s.sz = {0}; check("spinhalf L=12 sz=0", SpinHalfBasis(12, s), Idx::Lin, 12); }
        { Sector s; s.sz = {0.5}; check("spinhalf L=11 sz=0.5", SpinHalfBasis(11, s), Idx::Lin, 11); }
        { Sector s; s.sz = {0, 1}; check("spinhalf L=10 sz={0,1}", SpinHalfBasis(10, s), Idx::Bisect, 10); }
        { Sector s; s.n = {4}; check("fermion L=10 n=4", FermionBasis(10, s), Idx::Lin, 10); }
        { Sector s; s.nup = {2}; s.ndn = {3}; check("hubbard L=6 (2,3)", HubbardBasis(6, s), Idx::Lin, 12); }
        { Sector s; s.nup = {2}; s.ndn = {2}; check("tj L=6 (2,2)", TJBasis(6, s), Idx::Lin, 12, false); }

        // Identity index: the full space
        const Idx id = make_identity_index<Space>(1024);
        Index j = 0;
        assert(id.find(1023, j) && j == 1023);
        assert(!id.find(1024, j));
    }
    Kokkos::finalize();
    std::cout << "test_state_index passed\n";
    return 0;
}
