using Test
using QKrylov

@testset "QKrylov.jl" begin
    @testset "Sector" begin
        sec = Sector()
        @test sec.ptr != C_NULL
        set_sz!(sec, 0)
        set_hubbard_particles!(sec, 1, 1)
    end

    @testset "Site Types" begin
        s1 = SpinHalfSite()
        @test s1.ptr != C_NULL
        s2 = FermionSite()
        @test s2.ptr != C_NULL
        s3 = HubbardSite()
        @test s3.ptr != C_NULL
        s4 = TJSite()
        @test s4.ptr != C_NULL
    end

    @testset "Basis & Sectors" begin
        b_full = SpinHalfBasis(4)
        @test nsites(b_full) == 4
        @test dimension(b_full) == 16
        @test size(b_full) == (16, 16)

        sec = Sector()
        set_sz!(sec, 0)
        b_sec = SpinHalfBasis(4, sec)
        @test nsites(b_sec) == 4
        @test dimension(b_sec) == 6

        b_fermion = FermionBasis(4)
        @test dimension(b_fermion) == 16

        b_hubbard = HubbardBasis(2)
        @test dimension(b_hubbard) == 16

        b_tj = TJBasis(2)
        @test dimension(b_tj) == 9
    end

    @testset "OpSum & Hamiltonian Application" begin
        basis = SpinHalfBasis(2)
        site = SpinHalfSite()
        op = OpSum()
        
        # H = S^z_0 S^z_1 + 0.5 (S^+_0 S^-_1 + S^-_0 S^+_1)
        add_term!(op, 1.0, "Sz", 0, "Sz", 1)
        add_term!(op, 0.5, "Sp", 0, "Sm", 1)
        add_term!(op, 0.5, "Sm", 0, "Sp", 1)

        H = MatrixFreeHamiltonian(basis, site, op)
        @test dimension(H) == 4
        @test size(H) == (4, 4)

        x = [1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]
        y = H * x
        @test length(y) == 4
        @test isapprox(y[1], 0.25 + 0.0im, atol=1e-10)
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
        res = lanczos_ground_state(H, maxiter=50, tol=1e-12)
        
        # Ground state energy for 4-site periodic Heisenberg chain is -2.0
        @test isapprox(res.energy, -2.0, atol=1e-6)
    end
end
