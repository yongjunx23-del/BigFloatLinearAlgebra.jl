@testset "explicit threading" begin
    p = 256
    rng = MersenneTwister(9100)
    a = BigFloat(2; precision = p)
    b = BigFloat(3; precision = p)
    one = BigFloat(1; precision = p)
    threaded = BFLA.KernelConfig(thread_count = 4)

    @testset "gemm threaded bit-identical" begin
        A = random_matrix(32, 40, p, rng)
        B = random_matrix(40, 24, p, rng)
        C0 = random_matrix(32, 24, p, rng)
        Cs = BFLA.owned_copy(C0)
        BFLA.gemm!(Native, NoTrans, NoTrans, a, A, B, b, Cs)
        Ct = BFLA.owned_copy(C0)
        BFLA.gemm!(Native, NoTrans, NoTrans, a, A, B, b, Ct; config = threaded)
        @test all(Cs[i, j] == Ct[i, j] for i in 1:32, j in 1:24)
    end

    @testset "syrk threaded bit-identical" begin
        A = random_matrix(30, 20, p, rng)
        C0 = random_matrix(30, 30, p, rng)
        Cs = BFLA.owned_copy(C0)
        BFLA.syrk!(Native, Lower, NoTrans, a, A, b, Cs)
        Ct = BFLA.owned_copy(C0)
        BFLA.syrk!(Native, Lower, NoTrans, a, A, b, Ct; config = threaded)
        @test all(Cs[i, j] == Ct[i, j] for i in 1:30, j in 1:30)
    end

    @testset "trsm threaded multi-RHS bit-identical" begin
        n, r = 24, 16
        T = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        for j in 1:n, i in j:n
            T[i, j] = i == j ? BigFloat(n + i; precision = p) : BigFloat(rand(rng); precision = p)
        end
        B0 = random_matrix(n, r, p, rng)
        Bs = BFLA.owned_copy(B0)
        BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, T, Bs)
        Bt = BFLA.owned_copy(B0)
        BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, T, Bt; config = threaded)
        @test all(Bs[i, j] == Bt[i, j] for i in 1:n, j in 1:r)
    end
end
