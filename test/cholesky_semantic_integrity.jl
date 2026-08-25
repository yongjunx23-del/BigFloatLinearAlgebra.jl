# Cholesky semantic integrity (v0.2.2).
#
# A successful Cholesky factor must have a strictly positive diagonal. A caller
# that mutates a diagonal element to a non-positive value (keeping shape,
# precision, and finiteness intact) must be rejected by the checked solve and
# diagnostics instead of silently producing a wrong result.

using BigFloatLinearAlgebra
import MutableArithmetics as MA

@testset "cholesky semantic integrity" begin
    p = 128
    mk(x) = BigFloat(x; precision = p)

    A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
    for i in 1:3, j in 1:3
        A[i, j] = mk(i == j ? 4.0 : 1.0)
    end
    b = [mk(1), mk(2), mk(3)]

    @testset "ordinary Lower negative diagonal rejected" begin
        F = BFLA.cholesky(Native, A)
        @test issuccess(F)
        F.factors[2, 2] = mk(-1.0)
        @test_throws ArgumentError BFLA.solve!(F, b)
        @test_throws ArgumentError BFLA.ldiv!(F, b)
        @test_throws ArgumentError factor_diagnostics(F)
    end

    @testset "ordinary Lower zero diagonal rejected" begin
        F = BFLA.cholesky(Native, A)
        F.factors[3, 3] = mk(0.0)
        @test_throws ArgumentError BFLA.solve!(F, b)
        @test_throws ArgumentError factor_diagnostics(F)
    end

    @testset "cache Lower negative diagonal rejected" begin
        c = BFLACholeskyCache(Native)
        prepare!(c, 3, p)
        factorize!(c, A)
        c.factors[2, 2] = mk(-1.0)
        @test_throws ArgumentError solve!(BFLA.owned_zeros(BigFloat, 3; precision_bits = p), c, b)
        @test_throws ArgumentError solve(c, b)  # allocating solve
        @test_throws ArgumentError factor_diagnostics(c)
    end

    @testset "cache Lower zero diagonal rejected" begin
        c = BFLACholeskyCache(Native)
        prepare!(c, 3, p)
        factorize!(c, A)
        c.factors[1, 1] = mk(0.0)
        @test_throws ArgumentError solve(c, b)
        @test_throws ArgumentError factor_diagnostics(c)
    end

    @testset "normal factor still passes residual gate" begin
        F = BFLA.cholesky(Native, A)
        x = BFLA.solve(F, b)
        r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        BFLA.residual!(Native, NoTrans, A, x, b, r)
        @test BFLA.norminf(Native, r) <=
              max(100 * 3, 1) * eps_bits(p) * BFLA.norminf(Native, b)
    end

    @testset "Generic Upper normal factor passes" begin
        # A GenericBackend Upper Cholesky factor has a positive diagonal and
        # must continue to pass the checked solve.
        F = BFLA.cholesky(Generic, A; triangle = Upper)
        @test issuccess(F)
        @test factor_triangle(F) === Upper
        x = BFLA.solve(F, b)
        r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        BFLA.residual!(Generic, NoTrans, A, x, b, r)
        @test BFLA.norminf(Generic, r) <=
              max(100 * 3, 1) * eps_bits(p) * BFLA.norminf(Generic, b)
    end
end
