@testset "precision" begin
    @testset "Native ignores ambient global precision" begin
        p = 256
        rng = MersenneTwister(6000)
        A = random_matrix(4, 4, p, rng)
        B = random_matrix(4, 4, p, rng)
        C = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        # Force a much lower global precision; Native must not inherit it.
        setprecision(BigFloat, 64) do
            BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), C)
        end
        @test all(precision(x) == p for x in C)
        # compare against a reference computed at p bits
        Cref = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        BFLA.gemm!(Generic, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), Cref)
        assert_close(C, Cref, p; label = "precision independence")
    end

    @testset "explicit storage precision" begin
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 512)
        @test all(precision(x) == 512 for x in A)
        B = BFLA.owned_copy(A)
        @test all(precision(x) == 512 for x in B)
    end

    @testset "intra-array mixed precision fails closed" begin
        one256 = BigFloat(1; precision = 256)
        zero256 = BigFloat(0; precision = 256)
        for index in (1, 5, 9)  # first, middle, last linear element
            A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 256)
            A[index] = BigFloat(1; precision = 64)
            C = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 256)
            @test_throws BFLA.PrecisionMismatch BFLA.gemm!(
                Native, NoTrans, NoTrans, one256, A, A, zero256, C)
            @test_throws BFLA.PrecisionMismatch BFLA.cholesky!(Native, A; check = false)
            @test_throws BFLA.PrecisionMismatch BFLA.trsm!(
                Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one256, A, C)
        end

        # A uniformly 256-bit matrix must still be accepted (guard against an
        # over-eager scan that rejects homogeneous inputs).
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 256)
        C = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 256)
        @test BFLA.gemm!(Native, NoTrans, NoTrans, one256, A, A, zero256, C) === C
    end
end
