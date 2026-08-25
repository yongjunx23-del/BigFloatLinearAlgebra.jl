# Factor shape-integrity contract (v0.2.1).
#
# The checked `ldiv!`/`solve!`/`factor_diagnostics`/`factor_inertia`/
# `numerical_rank` paths must reject a malformed factor *shape* with a clear
# error before any `@inbounds` kernel runs. This covers rectangular factors
# where a square factor is required, cache factor storage that no longer
# matches the reserved order, and the RRQR ordinary-vs-cache shape rules.

using BigFloatLinearAlgebra

@testset "factor shape integrity" begin
    p = 128
    mk(x) = BigFloat(x; precision = p)

    # --- ordinary allocating factors: rectangular must be rejected ---------

    @testset "ordinary rectangular factors rejected" begin
        b = [mk(1), mk(2), mk(3)]
        rect = BFLA.owned_zeros(BigFloat, 2, 3; precision_bits = p)
        for i in 1:2, j in 1:3
            rect[i, j] = mk(1.0)
        end

        # Cholesky: a rectangular factor matrix must be rejected by the checked
        # solve and by factor_diagnostics.
        F = BFLA.BFLACholeskyFactor(rect, Native, Lower, p, BFLA.SUCCESS_STATUS)
        @test_throws DimensionMismatch BFLA.solve!(F, b)
        @test_throws DimensionMismatch BFLA.ldiv!(F, b)
        @test_throws DimensionMismatch factor_diagnostics(F)

        # LU: rectangular factor rejected.
        L = BFLA.BFLALUFactor(rect, Native, p, BFLA.SUCCESS_STATUS, collect(1:3), collect(1:3))
        @test_throws DimensionMismatch BFLA.solve!(L, b)
        @test_throws DimensionMismatch factor_diagnostics(L)

        # LDLT: rectangular factor rejected.
        D = BFLA.BFLALDLTFactor(rect, Native, p, BFLA.SUCCESS_STATUS, collect(1:3), [1, 1, 1], falses(3))
        @test_throws DimensionMismatch BFLA.solve!(D, b)
        @test_throws DimensionMismatch factor_diagnostics(D)
        @test_throws DimensionMismatch factor_inertia(D)
    end

    # --- ordinary RRQR: rectangular is legal ---------------------------------

    @testset "ordinary RRQR rectangular legal" begin
        # A 4x3 overdetermined system factors fine and solves in the least
        # squares sense. The returned vector has length m; the least-squares
        # solution occupies the first n entries.
        rng = MersenneTwister(7)
        A = random_matrix(4, 3, p, rng)
        F = BFLA.qr(Native, A)
        @test size(F) == (4, 3)
        b = random_vector(4, p, rng)
        x = BFLA.solve(F, b)
        @test length(x) == 4
        # Least-squares gate: the residual r = b - A*x[1:3] must be orthogonal
        # to the column space, i.e. ||A' * r|| is small relative to ||b||.
        xs = view(x, 1:3)
        r = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        BFLA.residual!(Native, NoTrans, A, xs, b, r)
        At = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        BFLA.gemv!(Native, Trans, one_p(p), A, r, BigFloat(0; precision = p), At)
        @test BFLA.norminf(Native, At) <=
              max(100 * 4, 1) * eps_bits(p) * BFLA.norminf(Native, b)
    end

    # --- cache factor storage shape -----------------------------------------

    @testset "cache factor storage shape" begin
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        for i in 1:3, j in 1:3
            A[i, j] = mk(i == j ? 4.0 : 1.0)
        end
        b = [mk(1), mk(2), mk(3)]

        for (make_cache, name) in (
            (() -> BFLACholeskyCache(Native), "Cholesky"),
            (() -> BFLALUCache(Native), "LU"),
            (() -> BFLALDLTCache(Native), "LDLT"),
            (() -> BFLARRQRCache(Native), "RRQR"),
        )
            @testset "$name cache wrong factor shape" begin
                cache = make_cache()
                prepare!(cache, 3, p)
                factorize!(cache, A)
                # Replace the owned factor storage with a wrong-size matrix.
                cache.factors = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
                @test_throws DimensionMismatch solve(cache, b)
                @test_throws DimensionMismatch solve!(BFLA.owned_zeros(BigFloat, 3; precision_bits = p), cache, b)
                @test_throws DimensionMismatch factor_diagnostics(cache)
            end
        end

        @testset "cache.n inconsistent with matrix shape" begin
            # A square-but-wrong-size factor (3x3) in a cache reserved for n=2
            # must be rejected even though it is square.
            cache = BFLACholeskyCache(Native)
            prepare!(cache, 2, p)
            cache.factors = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
            cache.n = 2
            cache.status = BFLA.FactorStatus(:success, nothing)
            @test_throws DimensionMismatch solve(cache, [mk(1), mk(2)])
        end

        @testset "RRQR cache rectangular explicitly rejected" begin
            # The RRQR cache is square-only; a rectangular factor storage must
            # be rejected even though ordinary RRQR allows rectangular.
            cache = BFLARRQRCache(Native)
            prepare!(cache, 3, p)
            factorize!(cache, A)
            cache.factors = BFLA.owned_zeros(BigFloat, 4, 3; precision_bits = p)
            @test_throws DimensionMismatch solve(cache, b)
            @test_throws DimensionMismatch factor_diagnostics(cache)
        end
    end

    # --- checked API must not enter an @inbounds kernel ----------------------

    @testset "no @inbounds kernel on malformed shape" begin
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        for i in 1:3, j in 1:3
            A[i, j] = mk(i == j ? 4.0 : 1.0)
        end
        b = [mk(1), mk(2), mk(3)]
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        cache.factors = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        # The error must be a clean DimensionMismatch, never a BoundsError.
        err = try
            solve(cache, b)
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch
        @test !(err isa BoundsError)
    end
end
