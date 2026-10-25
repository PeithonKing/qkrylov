// A single-sector SpinHalfBasis is a formula: size / state / index / contains must agree with a
// brute-force enumeration, and a Hamiltonian that generates its states itself (formula path) must
// give bit-identical results to one built from the materialized list (list path).
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <bit>
#include <cassert>
#include <iostream>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;
using Space = Kokkos::DefaultHostExecutionSpace;

static void check_basis(int N, double sz) {
    Sector sec; sec.sz = {sz};
    SpinHalfBasis b(N, sec);
    const int k = static_cast<int>(sz + N / 2.0 + 0.5);
    std::vector<StateID> brute;
    for (StateID s = 0; s < (StateID(1) << N); ++s) if (std::popcount(s) == k) brute.push_back(s);
    assert(b.size() == brute.size());
    int n = 0, kk = 0;
    assert(b.fixed_popcount_sector(n, kk) && n == N && kk == k);
    for (Index i = 0; i < b.size(); ++i) {
        assert(b.state(i) == brute[i]);
        assert(b.index(brute[i]) == i);
        assert(b.contains(brute[i]));
    }
    for (StateID s = 0; s < (StateID(1) << N); ++s) {
        if (std::popcount(s) != k) { assert(!b.contains(s)); bool threw = false; try { b.index(s); } catch (const std::runtime_error&) { threw = true; } assert(threw); }
    }
    bool threw = false;
    try { b.state(b.size()); } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    b.materialize();
    assert(!b.fixed_popcount_sector(n, kk));
    assert(b.state_list()->size() == brute.size());
    for (Index i = 0; i < b.size(); ++i) assert((*b.state_list())[i] == brute[i]);
}

static OpSum xxz(int L) {
    OpSum ops;
    for (int i = 0; i < L; ++i) {
        const int j = (i + 1) % L;
        ops.add_term(OperatorTerm(0.7, {{"Sz", i}, {"Sz", j}}));
        ops.add_term(OperatorTerm(0.5, {{"Sp", i}, {"Sm", j}}));
        ops.add_term(OperatorTerm(0.5, {{"Sm", i}, {"Sp", j}}));
        ops.add_term(OperatorTerm(0.1 * (i + 1), {{"Sz", i}}));
    }
    return ops;
}

static void check_hamiltonian(int N, double sz) {
    Sector sec; sec.sz = {sz};
    auto b_formula = std::make_shared<SpinHalfBasis>(N, sec);
    auto b_list = std::make_shared<SpinHalfBasis>(N, sec);
    b_list->materialize();
    auto site = std::make_shared<SpinHalfSite>();
    MatrixFreeHamiltonian<Space> Hf(b_formula, site, xxz(N)), Hl(b_list, site, xxz(N));
    assert(Hf.dimension() == Hl.dimension());
    VectorView<Space> x("x", Hf.dimension()), yf("yf", Hf.dimension()), yl("yl", Hf.dimension());
    for (Index i = 0; i < Hf.dimension(); ++i) x(i) = KComplex(std::sin(0.37 * i), std::cos(0.11 * i));
    Hf.apply(x, yf); Hl.apply(x, yl);
    for (Index i = 0; i < Hf.dimension(); ++i) { assert(yf(i).real() == yl(i).real() && yf(i).imag() == yl(i).imag()); }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        for (int N : {1, 2, 3, 6, 9, 12}) for (int k = 0; k <= N; ++k) check_basis(N, k - N / 2.0);
        for (int N : {4, 7, 12, 15, 16}) for (int k : {0, 1, N / 2, N - 1, N}) check_hamiltonian(N, k - N / 2.0);
    }
    Kokkos::finalize();
    std::cout << "test_fixed_popcount_basis passed\n";
    return 0;
}
