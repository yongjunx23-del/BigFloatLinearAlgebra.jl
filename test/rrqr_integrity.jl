# RRQR precision preflight and rank-policy semantic consistency (v0.2.1).
#
# `BFLARRQRCache.factorize!` must reject a rank-policy scalar whose precision
# differs from the cache precision, and the checked validator must reject
# in-range-but-wrong rank-policy corruption (rank, threshold, reference_scale,
# tolerance-vs-atol) that a pure range check would accept.

using BigFloatLinearAlgebra
import MutableArithmetics as MA

@testset "RRQR integrity" begin
    p = 128
    mk(x) = BigFloat(x; precision = p)

    A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
    for i in 1:3, j in 1:3
        A[i, j] = mk(i == j ? 4.0 : 1.0)
    end
    b = [mk(1), mk(2), mk(3)]

    @testset "factorize! precision preflight" begin
        cache = BFLARRQRCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        # A successful factor exists; a precision-mismatched rank-policy scalar
        # must be rejected and must preserve the previous factor.
        @test_throws BFLA.PrecisionMismatch factorize!(
            cache, A; atol = BigFloat(0; precision = 64),
        )
        @test_throws BFLA.PrecisionMismatch factorize!(
            cache, A; rtol = BigFloat(0; precision = 64),
        )
        @test_throws BFLA.PrecisionMismatch factorize!(
            cache, A; tol = BigFloat(0; precision = 64),
        )
        # Mixed precision among the tolerances is also rejected.
        @test_throws BFLA.PrecisionMismatch factorize!(
            cache, A; atol = BigFloat(0; precision = 64),
            rtol = BigFloat(0; precision = p),
        )
        # The previous successful factor is preserved.
        @test issuccess(cache)
        @test factor_rank(cache) >= 1
    end

    @testset "rank-policy semantic consistency" begin
        cache = BFLARRQRCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        @test factor_rank(cache) == 3
        # Save the original scalars so each corruption can be restored exactly
        # at the factor precision (avoiding `max`/`*` producing a higher
        # precision intermediate).
        orig_tol = MA.mutable_copy(cache.tolerance)
        orig_atol = MA.mutable_copy(cache.atol)
        orig_rtol = MA.mutable_copy(cache.rtol)
        orig_scale = MA.mutable_copy(cache.reference_scale)
        orig_eff = MA.mutable_copy(cache.effective_threshold)

        # tolerance must equal atol.
        cache.tolerance = mk(0.5)
        @test_throws ArgumentError solve(cache, b)
        @test_throws ArgumentError factor_diagnostics(cache)
        cache.tolerance = MA.mutable_copy(orig_tol)

        # effective_threshold must equal max(atol, rtol*reference_scale).
        cache.effective_threshold = mk(0.25)
        @test_throws ArgumentError solve(cache, b)
        @test_throws ArgumentError factor_diagnostics(cache)
        cache.effective_threshold = MA.mutable_copy(orig_eff)

        # reference_scale changed (finite, non-negative, but wrong).
        cache.reference_scale = mk(0.5)
        @test_throws ArgumentError solve(cache, b)
        @test_throws ArgumentError factor_diagnostics(cache)
        cache.reference_scale = MA.mutable_copy(orig_scale)

        # rank=0 (in range but wrong).
        cache.rank = 0
        @test_throws ArgumentError solve(cache, b)
        @test_throws ArgumentError factor_diagnostics(cache)
        cache.rank = 3

        # rank=n-1 (in range but wrong).
        cache.rank = 2
        @test_throws ArgumentError solve(cache, b)
        cache.rank = 3
    end

    @testset "rank-policy scalar precision" begin
        cache = BFLARRQRCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        # A 64-bit scalar in a 128-bit cache must be rejected.
        cache.atol = BigFloat(0; precision = 64)
        @test_throws BFLA.PrecisionMismatch solve(cache, b)
        @test_throws BFLA.PrecisionMismatch factor_diagnostics(cache)
        cache.atol = BigFloat(0; precision = p)

        cache.rtol = BigFloat(0; precision = 64)
        @test_throws BFLA.PrecisionMismatch solve(cache, b)
        cache.rtol = BigFloat(0; precision = p)
    end

    @testset "rank-policy scalar finiteness" begin
        cache = BFLARRQRCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        cache.atol = mk(NaN)
        @test_throws DomainError solve(cache, b)
        cache.atol = mk(0.0)

        cache.rtol = mk(Inf)
        @test_throws DomainError solve(cache, b)
        cache.rtol = mk(0.0)
    end

    @testset "trusted solve keeps caller contract" begin
        # The trusted path skips the factor rescan; a caller that guarantees the
        # factor is unchanged still gets a correct solve.
        cache = BFLARRQRCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        x = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        solve_trusted!(x, cache, b)
        r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        BFLA.residual!(Native, NoTrans, A, x, b, r)
        @test BFLA.norminf(Native, r) <=
              max(100 * 3, 1) * eps_bits(p) * BFLA.norminf(Native, b)
    end
end
