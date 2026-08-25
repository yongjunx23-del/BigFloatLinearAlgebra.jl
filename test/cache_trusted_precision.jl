# Cache trusted-solve precision contract (v0.2.2).
#
# `solve_trusted!` must reject a solution or RHS whose precision differs from
# the cache's factor precision, mirroring the ordinary `ldiv_trusted!`
# contract. The zero-Julia-allocation gate must be preserved after warm-up.

using BigFloatLinearAlgebra
import MutableArithmetics as MA

@testset "cache trusted precision contract" begin
    p = 256
    mk(x) = BigFloat(x; precision = p)

    A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
    for i in 1:3, j in 1:3
        A[i, j] = mk(i == j ? 4.0 : 1.0)
    end
    b128 = [BigFloat(1; precision = 128), BigFloat(2; precision = 128), BigFloat(3; precision = 128)]
    x128 = BFLA.owned_zeros(BigFloat, 3; precision_bits = 128)
    b256 = [mk(1), mk(2), mk(3)]
    x256 = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)

    for (ctor, name) in (
        (BFLACholeskyCache, "Cholesky"),
        (BFLALUCache, "LU"),
        (BFLALDLTCache, "LDLT"),
        (BFLARRQRCache, "RRQR"),
    )
        @testset "$name" begin
            c = ctor(Native)
            prepare!(c, 3, p)
            factorize!(c, A)

            # factor=256, x=128, b=128 -> PrecisionMismatch
            @test_throws BFLA.PrecisionMismatch solve_trusted!(x128, c, b128)
            # factor=256, x=256, b=128 -> PrecisionMismatch
            @test_throws BFLA.PrecisionMismatch solve_trusted!(x256, c, b128)
            # factor=256, x=128, b=256 -> PrecisionMismatch
            @test_throws BFLA.PrecisionMismatch solve_trusted!(x128, c, b256)
            # factor=256, x=256, b=256 -> success
            solve_trusted!(x256, c, b256)
            r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
            BFLA.residual!(Native, NoTrans, A, x256, b256, r)
            @test BFLA.norminf(Native, r) <=
                  max(100 * 3, 1) * eps_bits(p) * BFLA.norminf(Native, b256)
        end
    end

    @testset "matrix RHS precision contract" begin
        c = BFLALUCache(Native)
        prepare!(c, 3, p)
        factorize!(c, A)
        Bm128 = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = 128)
        Xm128 = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = 128)
        @test_throws BFLA.PrecisionMismatch solve_trusted!(Xm128, c, Bm128)
        Bm256 = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = p)
        Xm256 = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = p)
        solve_trusted!(Xm256, c, Bm256)
    end

    @testset "zero-allocation gate preserved" begin
        n = 16
        A16 = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        for j in 1:n, i in 1:n
            A16[i, j] = mk(i == j ? 4.0 : 1.0)
        end
        b16 = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
        for i in 1:n
            b16[i] = mk(i)
        end
        x16 = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
        for (ctor, name) in (
            (BFLACholeskyCache, "Cholesky"),
            (BFLALUCache, "LU"),
            (BFLALDLTCache, "LDLT"),
            (BFLARRQRCache, "RRQR"),
        )
            c = ctor(Native)
            prepare!(c, n, p)
            factorize!(c, A16)
            solve_trusted!(x16, c, b16)  # warm-up
            @test @allocated(solve_trusted!(x16, c, b16)) == 0
        end
    end
end
