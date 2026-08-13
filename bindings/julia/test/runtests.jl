using Test
using QuantumKrylov

@testset "QuantumKrylov.jl" begin
    @testset "Sector" begin
        sec = Sector()
        @test sec.ptr != C_NULL
        @test get_sz(sec) === nothing
        @test get_n(sec) === nothing
        @test string(sec) == "Sector(unconstrained)"

        set_sz!(sec, 0)
        @test get_sz(sec) == 0

        set_hubbard_particles!(sec, 1, 1)
        @test get_hubbard_particles(sec) == (1, 1)

        sec2 = Sector()
        set_n!(sec2, 2)
        @test get_n(sec2) == 2

        set_nb!(sec2, 1)
        @test get_nb(sec2) == 1
        @test string(sec2) == "Sector(N = 2, Nb = 1)"
    end

    @testset "Site Types" begin
        s1 = SpinHalfSite()
        @test s1.ptr != C_NULL
        @test string(s1) == "SpinHalfSite(dim = 2, states = [↑, ↓])"

        s2 = FermionSite()
        @test s2.ptr != C_NULL
        @test string(s2) == "FermionSite(dim = 2, states = [0, 1])"

        s3 = HubbardSite()
        @test s3.ptr != C_NULL
        @test string(s3) == "HubbardSite(dim = 4, states = [0, ↑, ↓, ↑↓])"

        s4 = TJSite()
        @test s4.ptr != C_NULL
        @test string(s4) == "TJSite(dim = 3, states = [0, ↑, ↓])"
    end

    @testset "Basis & Sectors & Lookups" begin
        b_full = SpinHalfBasis(4)
        @test nsites(b_full) == 4
        @test dimension(b_full) == 16
        @test size(b_full) == (16, 16)
        @test length(b_full) == 16
        @test string(b_full) == "SpinHalfBasis(sites = 4, dim = 16)"

        # State lookups
        st0 = state(b_full, 0)
        @test st0 == UInt64(0)
        @test b_full[1] == UInt64(0)
        @test basis_index(b_full, UInt64(0)) == 0
        @test UInt64(0) in b_full
        @test UInt64(15) in b_full

        sec = Sector()
        set_sz!(sec, 0)
        b_sec = SpinHalfBasis(4, sec)
        @test nsites(b_sec) == 4
        @test dimension(b_sec) == 6

        sec_f = Sector()
        set_n!(sec_f, 2)
        b_fermion = FermionBasis(4, sec_f)
        @test nsites(b_fermion) == 4
        @test dimension(b_fermion) == 6

        b_hubbard = HubbardBasis(2)
        @test dimension(b_hubbard) == 16

        b_tj = TJBasis(2)
        @test dimension(b_tj) == 9
    end

    @testset "OpSum, N-Body Terms & Hamiltonian Inspection" begin
        basis = SpinHalfBasis(3)
        site = SpinHalfSite()
        op = OpSum()

        # 1-body & 2-body terms
        add_term!(op, 1.0, "Sz", 0, "Sz", 1)
        # 3-body term: S^z_0 S^z_1 S^z_2
        add_term!(op, 0.5, ["Sz", "Sz", "Sz"], [0, 1, 2])

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 8

        # Extract diagonal
        diag_val = diagonal(H)
        @test length(diag_val) == 8

        x = zeros(ComplexF64, 8)
        x[8] = 1.0 + 0.0im # |111> spin down state
        y = H * x
        # Diagonal term for |111>: 1.0*(0.25) + 0.5*(0.125) = 0.3125
        @test isapprox(y[8], 0.3125 + 0.0im, atol=1e-6)
    end

    @testset "Lanczos Ground State Solver" begin
        # 4-site 1D Heisenberg chain
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()

        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end

        H = MatrixFreeHamiltonian(basis, site, op)

        # 1. Energy-only calculation (default)
        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        @test isapprox(res.energy, -2.0, atol=1e-6)
        @test_throws ErrorException res.state
        @test_throws ErrorException res.eigenvector

        # 2. State vector calculation
        res_state = lanczos_ground_state(H, maxiter=50, tol=1e-12, return_state=true)
        @test isapprox(res_state.energy, -2.0, atol=1e-6)
        psi = res_state.state
        @test length(psi) == 16
        @test res_state.eigenvector === psi

        # 3. Verify H * psi ≈ E0 * psi
        H_psi = H * psi
        @test isapprox(H_psi, res_state.energy .* psi, atol=1e-5)

        # 4. Destructuring test
        E0, psi_destruct = lanczos_ground_state(H, return_state=true)
        @test isapprox(E0, -2.0, atol=1e-6)
        @test length(psi_destruct) == 16
    end

    @testset "Davidson Solver" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        res = davidson_lowest(H, n_eig=2, max_subspace=10, tol=1e-6)
        @test length(res.eigenvalues) == 2
        @test isapprox(res.eigenvalues[1], -2.0, atol=1e-5)
        @test res.eigenvectors !== nothing
        @test length(res.eigenvectors) == 2
        @test length(res.eigenvectors[1]) == 16
    end

    @testset "Dynamics & Spectral Function" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        phi0 = zeros(ComplexF64, 16)
        phi0[1] = 1.0

        cfr = continued_fraction_coeffs(H, phi0, n_iter=10)
        @test length(cfr.alphas) > 0
        @test length(cfr.betas) >= 0

        spec_val = evaluate_spectral_function(cfr, 0.5, -2.0, 0.1)
        @test spec_val >= 0.0
    end

    @testset "FTLM Solver" begin
        N = 4
        basis = SpinHalfBasis(N)
        site = SpinHalfSite()
        op = OpSum()
        for i in 0:(N-1)
            next_i = mod(i + 1, N)
            add_term!(op, 1.0, "Sz", i, "Sz", next_i)
            add_term!(op, 0.5, "Sp", i, "Sm", next_i)
            add_term!(op, 0.5, "Sm", i, "Sp", next_i)
        end
        H = MatrixFreeHamiltonian(basis, site, op)

        ftlm_res = ftlm(H, beta=1.0, n_random=5, n_steps=20)
        @test isapprox(ftlm_res.beta, 1.0)
        @test ftlm_res.partition_function > 0.0
    end
end
