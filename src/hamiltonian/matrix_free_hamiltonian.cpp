#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include <stdexcept>
#include <algorithm>
#include <bit>
#include <string>
#include <type_traits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

// Safe, fast, portable popcount for Kokkos device code
KOKKOS_INLINE_FUNCTION int qkrylov_popcount64(uint64_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

// Index of the lowest set bit (x != 0), portable across host and device.
KOKKOS_INLINE_FUNCTION int qkrylov_ctz64(uint64_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ffsll(static_cast<long long>(x)) - 1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0; while (!(x & 1ULL)) { x >>= 1; ++n; } return n;
#endif
}


template <typename ExecSpace>
MatrixFreeHamiltonian<ExecSpace>::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops,
    Device device
)
    : basis_(std::move(basis)),
      site_(std::move(site)),
      ops_(ops),
      device_(std::move(device))
{
    init();
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::init()
{
    if (!basis_ || !site_) {
        throw std::invalid_argument("MatrixFreeHamiltonian requires both Basis and Site.");
    }
    if (basis_->bits_per_site() != site_->bits_per_site()) {
        throw std::invalid_argument(
            "Basis and Site dimension mismatch (bits_per_site: " + 
            std::to_string(basis_->bits_per_site()) + " vs " + 
            std::to_string(site_->bits_per_site()) + "). " +
            "You are mixing incompatible physics rules."
        );
    }

    detail::ensure_kokkos_initialized();
    real_coeffs_ = true;
    dim_ = basis_->size();
    if (dim_ == 0) return;

    // 1. State list -> device, plus the state->index lookup (no virtual calls in kernels).
    //    Identity bases (full spin-1/2 space) store nothing: state == index.
    identity_basis_ = basis_->is_identity();
    int fp_nbits = 0, fp_k = 0;
    bool generated = false;
    if (!identity_basis_ && basis_->fixed_popcount_sector(fp_nbits, fp_k)) {
        // A single fixed-popcount sector: lookup tables come from binomial formulas and the
        // state list is written in parallel on this Hamiltonian's device -- the basis never
        // builds (or sorts, or uploads) an O(dim) list.
        generated = build_fixed_popcount_index<ExecSpace>(fp_nbits, fp_k, index_, basis_states_);
    }
    if (!generated) {
        const std::vector<StateID>* list = basis_->state_list();
        std::vector<StateID> owned;
        if (!identity_basis_ && !list) {
            owned.resize(dim_);
            for (Index i = 0; i < dim_; ++i) owned[i] = basis_->state(i);
            list = &owned;
        }
        if (!identity_basis_) {
            identity_basis_ = true;
            for (Index i = 0; i < dim_; ++i) {
                if ((*list)[i] != static_cast<StateID>(i)) { identity_basis_ = false; break; }
            }
        }
        if (identity_basis_) {
            basis_states_ = Kokkos::View<StateID*, ExecSpace>();
            index_ = make_identity_index<ExecSpace>(dim_);
        } else {
            if (std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace> && list != &owned) {
                // Host execution: alias the basis' own state vector instead of copying it
                // (saves 8 bytes per state; H keeps the basis alive via basis_).
                basis_states_ = Kokkos::View<StateID*, ExecSpace, Kokkos::MemoryUnmanaged>(
                    const_cast<StateID*>(list->data()), dim_);
            } else {
                basis_states_ = Kokkos::View<StateID*, ExecSpace>(
                    Kokkos::view_alloc("qkrylov::basis_states", Kokkos::WithoutInitializing), dim_);
                Kokkos::View<const StateID*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(list->data(), dim_);
                Kokkos::deep_copy(basis_states_, h);
            }
            index_ = build_state_index<ExecSpace>(*list, basis_->conserved_masks(), basis_states_);
        }
    }

    // 2. Compile VM instructions on host and split diag/offdiag
    std::vector<Instruction> diag_insts;
    std::vector<Instruction> offdiag_insts;
    for (const auto& term : ops_.terms()) {
        auto compiled = site_->compile(term);
        for (const auto& inst : compiled) {
            bool is_diagonal;
            if (inst.based.d == 0) {
                is_diagonal = (inst.bin.flip_mask == 0);
            } else {
                is_diagonal = true;
                for(int k=0; k<inst.based.num_sites; ++k) {
                    if (inst.based.src_sigma[k] != inst.based.tgt_sigma[k]) { is_diagonal = false; break; }
                }
            }
            if (is_diagonal) {
                diag_insts.push_back(inst);
            } else {
                offdiag_insts.push_back(inst);
            }
        }
    }

    // 2b. Merge binary off-diagonal pairs that differ only in the pattern of their flipped
    //     bits (e.g. Sp_i Sm_j and Sm_i Sp_j): one instruction, valid for either pattern.
    {
        std::vector<Instruction> merged;
        std::vector<bool> used(offdiag_insts.size(), false);
        for (size_t a = 0; a < offdiag_insts.size(); ++a) {
            if (used[a]) continue;
            Instruction ia = offdiag_insts[a];
            if (ia.based.d == 0 && (ia.bin.flip_mask & ~ia.check_mask) == 0) {
                for (size_t b = a + 1; b < offdiag_insts.size(); ++b) {
                    const Instruction& ib = offdiag_insts[b];
                    if (used[b] || ib.based.d != 0) continue;
                    if (ib.bin.flip_mask == ia.bin.flip_mask && ib.check_mask == ia.check_mask &&
                        ib.sign_mask == ia.sign_mask && ib.coeff == ia.coeff &&
                        (ib.expected_bits ^ ia.expected_bits) == ia.bin.flip_mask) {
                        ia.bin.alt_xor = ia.bin.flip_mask;
                        used[b] = true;
                        break;
                    }
                }
            }
            merged.push_back(ia);
        }
        offdiag_insts = std::move(merged);
    }
    for (const auto& inst : offdiag_insts) {
        if (inst.coeff.imag() != 0.0) real_coeffs_ = false;
    }
    for (const auto& inst : diag_insts) {
        if (inst.coeff.imag() != 0.0) real_coeffs_ = false;
    }

    // 3. Setup and execute Diagonal VM on device
    int num_diag = diag_insts.size();
    Kokkos::View<Instruction*, ExecSpace> d_diag_inst("diag_inst", num_diag);
    if (num_diag > 0) {
        auto h_diag_inst = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(diag_insts.data(), num_diag);
        Kokkos::deep_copy(ExecSpace(), d_diag_inst, h_diag_inst);
    }
    
    // Real Hamiltonians keep only a Real diagonal (half the memory and traffic).
    if (real_coeffs_) {
        diagonal_real_ = RealVectorView<ExecSpace>("qkrylov::diagonal_real", dim_);
        diagonal_ = VectorView<ExecSpace>();
    } else {
        diagonal_ = VectorView<ExecSpace>("qkrylov::diagonal", dim_);
        diagonal_real_ = RealVectorView<ExecSpace>();
    }
    auto diag_view = diagonal_;
    auto diag_real_view = diagonal_real_;
    const bool real_diag = real_coeffs_;
    auto states_view = basis_states_;
    const bool identity = identity_basis_;

    Kokkos::parallel_for("qkrylov::init_diagonal",
        Kokkos::RangePolicy<ExecSpace, Index>(0, dim_),
        KOKKOS_LAMBDA(const Index alpha) {
            const StateID state = identity ? static_cast<StateID>(alpha) : states_view(alpha);
            double diag_r = 0.0;
            double diag_i = 0.0;
            for (int i = 0; i < num_diag; ++i) {
                const auto& inst = d_diag_inst(i);
                // Binary: bitmask check. Base-d: modular residue check.
                if (inst.based.d == 0) {
                    if ((state & inst.check_mask) != inst.expected_bits) continue;
                } else {
                    uint64_t d = inst.based.d;
                    bool valid = true;
                    for(int k=0; k<inst.based.num_sites; ++k) {
                        uint64_t d_pow = 1; for(int p=0; p<inst.based.site[k]; ++p) d_pow *= d;
                        uint64_t local = (state / d_pow) % d;
                        if (local != inst.based.tgt_sigma[k]) { valid = false; break; }
                    }
                    if (!valid) continue;
                }

                double real_c = reinterpret_cast<const double*>(&inst.coeff)[0];
                double imag_c = reinterpret_cast<const double*>(&inst.coeff)[1];
                
                if (inst.sign_mask) {
                    if (qkrylov_popcount64(state & inst.sign_mask) % 2 != 0) {
                        real_c = -real_c;
                        imag_c = -imag_c;
                    }
                }
                diag_r += real_c;
                diag_i += imag_c;
            }
            if (real_diag) diag_real_view(alpha) = static_cast<Real>(diag_r);
            else diag_view(alpha) = KComplex(static_cast<Real>(diag_r), static_cast<Real>(diag_i));
        }
    );

    // 4. Store off-diagonal VM instructions
    num_instructions_ = offdiag_insts.size();
    if (num_instructions_ > 0) {
        instructions_ = Kokkos::View<Instruction*, ExecSpace>("qkrylov::instructions", num_instructions_);
        auto host_view = Kokkos::View<const Instruction*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(offdiag_insts.data(), num_instructions_);
        Kokkos::deep_copy(ExecSpace(), instructions_, host_view);
    }

    // 5. Flat hot-path arrays (binary instructions only; base-d keeps the generic kernel).
    all_binary_ = true;
    for (const auto& inst : offdiag_insts) {
        if (inst.based.d != 0) { all_binary_ = false; break; }
    }
    if (all_binary_) {
        const size_t n = offdiag_insts.size();
        std::vector<uint64_t> flip(n), check(n), expected(n), alt(n), sign(n);
        std::vector<Real> c_re(n), c_im(n);
        for (size_t i = 0; i < n; ++i) {
            const auto& in = offdiag_insts[i];
            flip[i] = in.bin.flip_mask; check[i] = in.check_mask; expected[i] = in.expected_bits;
            alt[i] = in.bin.alt_xor; sign[i] = in.sign_mask;
            c_re[i] = static_cast<Real>(in.coeff.real()); c_im[i] = static_cast<Real>(in.coeff.imag());
        }
        auto up = [](const auto& v, const char* label) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            Kokkos::View<T*, ExecSpace> d(std::string(label), v.size());
            Kokkos::deep_copy(d, Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
            return d;
        };
        // "Safe" instructions: whenever the source pattern matches, the hop provably lands
        // inside the basis, so the kernel can skip the bounds and class checks.
        //  - Identity basis: safe if all flipped bits are below the dimension (2^N).
        //  - Lin basis: safe if the hop keeps every conserved popcount (as many 1->0 as 0->1
        //    flips inside each mask; all flipped bits constrained by the pattern) and stays
        //    below the table's bit width.
        std::vector<uint8_t> safe(n, 0);
        {
            const auto masks = basis_->conserved_masks();
            for (size_t i = 0; i < n; ++i) {
                const auto& in = offdiag_insts[i];
                const uint64_t f = in.bin.flip_mask;
                if (index_.mode == StateIndex<ExecSpace>::Identity) {
                    safe[i] = (f < static_cast<uint64_t>(dim_)) ? 1 : 0;
                } else if (index_.mode == StateIndex<ExecSpace>::Lin && index_.complete) {
                    const int nbits = index_.low_bits + std::bit_width(index_.n_hi - 1);
                    bool ok = (f & ~in.check_mask) == 0 && (nbits >= 64 || (f >> nbits) == 0);
                    for (uint64_t m : masks) {
                        if (std::popcount(f & m & in.expected_bits) != std::popcount(f & m & ~in.expected_bits)) ok = false;
                    }
                    safe[i] = ok ? 1 : 0;
                }
            }
        }
        i_safe_ = up(safe, "qkrylov::i_safe");

        // Byte-lane validity tables (see header). An instruction is lane-local when its
        // check_mask is non-empty and lies inside one byte of the state.
        n_chunks_ = static_cast<int>((n + 63) / 64);
        int max_lane = 0;
        std::vector<int> lane_of(n, -1);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t cm = check[i];
            if (cm == 0) continue;
            const int lo = std::countr_zero(cm) / 8, hi = (63 - std::countl_zero(cm)) / 8;
            if (lo == hi) { lane_of[i] = lo; max_lane = std::max(max_lane, lo); }
        }
        n_lanes_ = max_lane + 1;
        std::vector<uint64_t> tab(static_cast<size_t>(n_chunks_) * n_lanes_ * 256, 0);
        std::vector<int32_t> gen, gen_start(n_chunks_ + 1, 0);
        for (int c = 0; c < n_chunks_; ++c) {
            gen_start[c] = static_cast<int32_t>(gen.size());
            for (size_t i = static_cast<size_t>(c) * 64; i < std::min(n, static_cast<size_t>(c + 1) * 64); ++i) {
                const int k = lane_of[i];
                if (k < 0) { gen.push_back(static_cast<int32_t>(i)); continue; }
                const int bitpos = static_cast<int>(i - static_cast<size_t>(c) * 64);
                for (uint64_t v = 0; v < 256; ++v) {
                    const uint64_t sb = v << (8 * k);
                    const uint64_t d = (((sb ^ flip[i]) & check[i]) ^ expected[i]);
                    if (d == 0 || d == alt[i]) {
                        tab[(static_cast<size_t>(c) * n_lanes_ + k) * 256 + v] |= 1ULL << bitpos;
                    }
                }
            }
        }
        gen_start[n_chunks_] = static_cast<int32_t>(gen.size());
        if (gen.empty()) gen.push_back(0);
        lane_tab_ = up(tab, "qkrylov::lane_tab");
        gen_idx_ = up(gen, "qkrylov::gen_idx");
        gen_start_ = up(gen_start, "qkrylov::gen_start");
        i_flip_ = up(flip, "qkrylov::i_flip"); i_check_ = up(check, "qkrylov::i_check");
        i_expected_ = up(expected, "qkrylov::i_expected"); i_alt_ = up(alt, "qkrylov::i_alt");
        i_sign_ = up(sign, "qkrylov::i_sign");
        i_coeff_re_ = up(c_re, "qkrylov::i_coeff_re"); i_coeff_im_ = up(c_im, "qkrylov::i_coeff_im");
    }
}

template <typename ExecSpace>
const VectorView<ExecSpace>& MatrixFreeHamiltonian<ExecSpace>::diagonal() const
{
    if (real_coeffs_ && diagonal_.extent(0) != dim_) {
        diagonal_ = VectorView<ExecSpace>("qkrylov::diagonal", dim_);
        auto dc = diagonal_;
        auto dr = diagonal_real_;
        Kokkos::parallel_for("qkrylov::diagonal_to_complex",
            Kokkos::RangePolicy<ExecSpace, Index>(0, dim_),
            KOKKOS_LAMBDA(const Index i) { dc(i) = KComplex(dr(i), Real(0.0)); });
    }
    return diagonal_;
}

/// One row of y = H x (pull model: each row gathers from its source states, no atomics).
/// Scalar is KComplex, or Real for Hamiltonians whose coefficients are all real.
template <typename ExecSpace, typename Scalar>
struct HApplyRowT {
    VectorView<ExecSpace> diag;
    RealVectorView<ExecSpace> diag_r;
    Kokkos::View<qkrylov::Instruction*, ExecSpace> insts;
    int num_inst;
    Kokkos::View<StateID*, ExecSpace> states;
    StateIndex<ExecSpace> index;
    bool identity;
    bool real_coeffs;
    Kokkos::View<Scalar*, typename ExecSpace::memory_space> x;

    KOKKOS_INLINE_FUNCTION Scalar operator()(const Index beta) const {
        constexpr bool is_real = std::is_same_v<Scalar, Real>;
        const StateID state_beta = identity ? static_cast<StateID>(beta) : states(beta);

        // 1. Diagonal contribution
        Scalar y_val;
        if constexpr (is_real) y_val = diag_r(beta) * x(beta);
        else y_val = real_coeffs ? diag_r(beta) * x(beta) : diag(beta) * x(beta);

        // 2. Off-diagonal VM execution
        for (int i = 0; i < num_inst; ++i) {
            const auto& inst = insts(i);

            StateID state_alpha;
            if (inst.based.d == 0) {
                // Binary site: reverse XOR to find source state alpha
                state_alpha = state_beta ^ inst.bin.flip_mask;
                const uint64_t diff = (state_alpha & inst.check_mask) ^ inst.expected_bits;
                if (diff != 0 && diff != inst.bin.alt_xor) continue;
            } else {
                // Explicit base-d loop:
                uint64_t d = inst.based.d;
                bool valid = true;
                state_alpha = state_beta;
                for (int k = 0; k < inst.based.num_sites; ++k) {
                    uint64_t d_pow = 1; for (int p = 0; p < inst.based.site[k]; ++p) d_pow *= d;
                    uint64_t local = (state_beta / d_pow) % d;
                    // For off-diagonal, state_beta has tgt_sigma. For diagonal, src==tgt.
                    if (local != inst.based.tgt_sigma[k]) { valid = false; break; }
                    state_alpha = state_alpha - inst.based.tgt_sigma[k] * d_pow + inst.based.src_sigma[k] * d_pow;
                }
                if (!valid) continue;
            }

            Index alpha;
            if (!index.find(state_alpha, alpha)) continue;

            double real_c = reinterpret_cast<const double*>(&inst.coeff)[0];
            double imag_c = reinterpret_cast<const double*>(&inst.coeff)[1];

            if (inst.sign_mask) {
                if (qkrylov_popcount64(state_alpha & inst.sign_mask) % 2 != 0) {
                    real_c = -real_c;
                    imag_c = -imag_c;
                }
            }

            if constexpr (is_real) {
                y_val += static_cast<Real>(real_c) * x(alpha);
            } else if (real_coeffs) {
                y_val += static_cast<Real>(real_c) * x(alpha);
            } else {
                y_val += KComplex(static_cast<Real>(real_c), static_cast<Real>(imag_c)) * x(alpha);
            }
        }
        return y_val;
    }
};

/// Hot-path row kernel: binary instructions only, index mode fixed at compile time,
/// instructions as flat arrays with precision-converted coefficients. (A branch-free
/// variant that multiplies invalid terms by 0 was measured slower: the extra lookups
/// and gathers cost more than the mispredicted branches.)
template <typename ExecSpace, typename Scalar, int Mode>
struct FastRow {
    using Mem = typename ExecSpace::memory_space;
    VectorView<ExecSpace> diag;
    RealVectorView<ExecSpace> diag_r;
    bool real_coeffs;
    Kokkos::View<uint64_t*, ExecSpace> flip, check, expected, alt, sign;
    Kokkos::View<uint8_t*, ExecSpace> safe;
    Kokkos::View<Real*, ExecSpace> c_re, c_im;
    int n;
    Kokkos::View<StateID*, ExecSpace> states;
    StateIndex<ExecSpace> index;
    Kokkos::View<Scalar*, Mem> x;
    int n_lanes;
    Kokkos::View<uint64_t*, ExecSpace> lane_tab;
    Kokkos::View<int32_t*, ExecSpace> gen_idx, gen_start;

    KOKKOS_INLINE_FUNCTION Scalar operator()(const Index beta) const {
        constexpr bool is_real = std::is_same_v<Scalar, Real>;
        constexpr bool identity = (Mode == StateIndex<ExecSpace>::Identity);
        const StateID sb = identity ? static_cast<StateID>(beta) : states(beta);

        Scalar y;
        if constexpr (is_real) y = diag_r(beta) * x(beta);
        else y = real_coeffs ? diag_r(beta) * x(beta) : diag(beta) * x(beta);

        // Pass 1 (branch-free): which instructions apply to this row? One bit each, in
        // chunks of 64. Pass 2 walks only the set bits, so the unpredictable "is this hop
        // allowed" decision never turns into a mispredicted branch.
        for (int base = 0; base < n; base += 64) {
            const int cnt = (n - base) < 64 ? (n - base) : 64;
            (void)cnt;
            const int c = base >> 6;
            uint64_t valid = 0;
            const uint64_t* tab = &lane_tab((static_cast<size_t>(c) * n_lanes) * 256);
            for (int k = 0; k < n_lanes; ++k) {
                valid |= tab[k * 256 + ((sb >> (8 * k)) & 255)];
            }
            for (int g = gen_start(c); g < gen_start(c + 1); ++g) {
                const int i = gen_idx(g);
                const StateID sa = sb ^ flip(i);
                const uint64_t d = (sa & check(i)) ^ expected(i);
                valid |= static_cast<uint64_t>((d == 0) | (d == alt(i))) << (i - base);
            }
            while (valid) {
                const int i = base + qkrylov_ctz64(valid);
                valid &= valid - 1;
                const StateID sa = sb ^ flip(i);
                Index ia;
                const bool trusted = safe(i) != 0;
                if constexpr (identity) {
                    if (!trusted && sa >= static_cast<StateID>(index.dim)) continue;
                    ia = static_cast<Index>(sa);
                } else {
                    const uint64_t h = sa >> index.low_bits;
                    const uint64_t l = sa & index.low_mask;
                    if (!trusted) {
                        if (h >= index.n_hi) continue;
                        const int32_t c = index.lo_class(l);
                        if (c < 0 || c != index.hi_class(h)) continue;
                    }
                    ia = index.hi_off(h) + index.lo_rank(l);
                }
                const uint64_t sm = sign(i);
                const bool neg = sm != 0 && (qkrylov_popcount64(sa & sm) & 1);
                const Real cr = neg ? -c_re(i) : c_re(i);
                if constexpr (is_real) {
                    y += cr * x(ia);
                } else if (real_coeffs) {
                    y += cr * x(ia);
                } else {
                    y += KComplex(cr, neg ? -c_im(i) : c_im(i)) * x(ia);
                }
            }
        }
        return y;
    }
};

// Kernel launchers as named functors (not generic lambdas): NVCC forbids extended
// lambdas inside generic lambdas, so dispatch passes the row type through these.
template <typename ExecSpace, typename Scalar>
struct ApplyOp {
    Kokkos::View<Scalar*, typename ExecSpace::memory_space> y;
    Index dim;
    template <typename Row> void operator()(const Row& row) const {
        auto yy = y;
        Kokkos::parallel_for("qkrylov::H_apply", Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
            KOKKOS_LAMBDA(const Index b) { yy(b) = row(b); });
    }
};

template <typename ExecSpace, typename Scalar>
struct ApplyDotOp {
    using Acc = std::conditional_t<std::is_same_v<Scalar, Real>, double, Kokkos::complex<double>>;
    Kokkos::View<Scalar*, typename ExecSpace::memory_space> x, y;
    Index dim;
    Acc* out;
    template <typename Row> void operator()(const Row& row) const {
        auto xx = x; auto yy = y;
        Acc result{};
        Kokkos::parallel_reduce("qkrylov::H_apply_dot", Kokkos::RangePolicy<ExecSpace, Index>(0, dim),
            KOKKOS_LAMBDA(const Index b, Acc& sum) {
                const Scalar v = row(b);
                yy(b) = v;
                sum += dot_term(xx(b), v);
            },
            result);
        *out = result;
    }
};

template <typename ExecSpace>
template <typename Scalar, typename Op>
void MatrixFreeHamiltonian<ExecSpace>::dispatch_rows(
    const Kokkos::View<Scalar*, typename ExecSpace::memory_space>& x, const Op& op) const
{
    using SI = StateIndex<ExecSpace>;
    auto fast = [&](auto mode_tag) {
        constexpr int M = decltype(mode_tag)::value;
        op(FastRow<ExecSpace, Scalar, M>{diagonal_, diagonal_real_, real_coeffs_,
            i_flip_, i_check_, i_expected_, i_alt_, i_sign_, i_safe_, i_coeff_re_, i_coeff_im_,
            num_instructions_, basis_states_, index_, x,
            n_lanes_, lane_tab_, gen_idx_, gen_start_});
    };
    if (all_binary_ && index_.mode == SI::Identity) {
        fast(std::integral_constant<int, SI::Identity>{});
    } else if (all_binary_ && index_.mode == SI::Lin) {
        fast(std::integral_constant<int, SI::Lin>{});
    } else {
        op(HApplyRowT<ExecSpace, Scalar>{diagonal_, diagonal_real_, instructions_, num_instructions_,
            basis_states_, index_, identity_basis_, real_coeffs_, x});
    }
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    if (dim_ == 0) return;
    // y is fully overwritten (one write per row), so no zero-fill pass.
    dispatch_rows<KComplex>(x, ApplyOp<ExecSpace, KComplex>{y, dim_});
}

template <typename ExecSpace>
KComplex MatrixFreeHamiltonian<ExecSpace>::apply_dot(
    const VectorView<ExecSpace>& x,
    VectorView<ExecSpace>& y
) const
{
    if (dim_ == 0) return KComplex(0.0, 0.0);
    Kokkos::complex<double> r(0.0, 0.0);
    dispatch_rows<KComplex>(x, ApplyDotOp<ExecSpace, KComplex>{x, y, dim_, &r});
    return KComplex(static_cast<Real>(r.real()), static_cast<Real>(r.imag()));
}

template <typename ExecSpace>
Real MatrixFreeHamiltonian<ExecSpace>::apply_dot_real(
    const RealVectorView<ExecSpace>& x,
    RealVectorView<ExecSpace>& y
) const
{
    if (!real_coeffs_) {
        throw std::logic_error("apply_dot_real requires a Hamiltonian with real coefficients");
    }
    if (dim_ == 0) return Real(0.0);
    double r = 0.0;
    dispatch_rows<Real>(x, ApplyDotOp<ExecSpace, Real>{x, y, dim_, &r});
    return static_cast<Real>(r);
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply_real(
    const RealVectorView<ExecSpace>& x,
    RealVectorView<ExecSpace>& y
) const
{
    if (!real_coeffs_) {
        throw std::logic_error("apply_real requires a Hamiltonian with real coefficients");
    }
    if (dim_ == 0) return;
    dispatch_rows<Real>(x, ApplyOp<ExecSpace, Real>{y, dim_});
}

template <typename ExecSpace>
void MatrixFreeHamiltonian<ExecSpace>::apply(
    const Complex* x,
    Complex* y
) const
{
    const Index dim = dim_;
    if (dim == 0) return;

    // Host execution: run directly on the caller's arrays (no scratch vectors, no copies).
    // y must not alias x because the kernel gathers from x while writing y.
    if constexpr (std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>) {
        if (static_cast<const void*>(x) != static_cast<const void*>(y)) {
            VectorView<ExecSpace> xv = Kokkos::View<KComplex*, ExecSpace, Kokkos::MemoryUnmanaged>(
                const_cast<KComplex*>(reinterpret_cast<const KComplex*>(x)), dim);
            VectorView<ExecSpace> yv = Kokkos::View<KComplex*, ExecSpace, Kokkos::MemoryUnmanaged>(
                reinterpret_cast<KComplex*>(y), dim);
            apply(xv, yv);
            return;
        }
    }

    if (scratch_x_.extent(0) != dim) {
        scratch_x_ = VectorView<ExecSpace>("scratch_x", dim);
        scratch_y_ = VectorView<ExecSpace>("scratch_y", dim);
    }

    auto x_host = Kokkos::View<const KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<const KComplex*>(x), dim);
    Kokkos::deep_copy(ExecSpace(), scratch_x_, x_host);

    apply(scratch_x_, scratch_y_);

    auto y_host = Kokkos::View<KComplex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<KComplex*>(y), dim);
    Kokkos::deep_copy(ExecSpace(), y_host, scratch_y_);
}

template <typename ExecSpace>
HostVector MatrixFreeHamiltonian<ExecSpace>::diagonal_host() const
{
    HostVector result;
    copy_device_to_host(diagonal(), result);
    return result;
}

#ifdef KOKKOS_ENABLE_SERIAL
template class MatrixFreeHamiltonian<Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class MatrixFreeHamiltonian<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class MatrixFreeHamiltonian<Kokkos::Threads>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class MatrixFreeHamiltonian<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_HIP
template class MatrixFreeHamiltonian<Kokkos::HIP>;
#endif
#ifdef KOKKOS_ENABLE_SYCL
template class MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>;
#endif

} // namespace QKRYLOV_PRECISION_NAMESPACE
} // namespace qkrylov
