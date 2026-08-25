# Allocating `solve(cache, b)` must go through the checked
# `_solve_owned_checked!` entry, which validates factor shape/storage/metadata
# and RHS finiteness before solving. It must never silently bypass the checked
# contract (the pre-v0.2.1 path called `solve_trusted!` and skipped factor
# integrity validation).

using BigFloatLinearAlgebra

@testset "cache allocating solve integrity" begin
    p = 128
    mk(x) = BigFloat(x; precision = p)

    A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
    for i in 1:3, j in 1:3
        A[i, j] = mk(i == j ? 4.0 : 1.0)
    end
    b = [mk(1), mk(2), mk(3)]

    @testset "normal allocating solve result" begin
        for (make_cache, name) in (
            (() -> BFLACholeskyCache(Native), "Cholesky"),
            (() -> BFLALUCache(Native), "LU"),
            (() -> BFLALDLTCache(Native), "LDLT"),
            (() -> BFLARRQRCache(Native), "RRQR"),
        )
            @testset "$name" begin
                cache = make_cache()
                prepare!(cache, 3, p)
                factorize!(cache, A)
                x = solve(cache, b)
                @test length(x) == 3
                # residual gate: ||A x - b|| small relative to ||b||
                r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
                BFLA.residual!(Native, NoTrans, A, x, b, r)
                @test BFLA.norminf(Native, r) <=
                      max(100 * 3, 1) * eps_bits(p) * BFLA.norminf(Native, b)
            end
        end
    end

    @testset "corrupted metadata rejected" begin
        # LU pivots corrupted.
        lc = BFLALUCache(Native)
        prepare!(lc, 3, p)
        factorize!(lc, A)
        lc.pivots[1] = 5
        @test_throws ArgumentError solve(lc, b)

        # LDLT blocks corrupted (finite but inconsistent: sum 4 != 3).
        dc = BFLALDLTCache(Native)
        prepare!(dc, 3, p)
        factorize!(dc, A)
        dc.blocks = [1, 1, 1, 1]
        @test_throws ArgumentError solve(dc, b)

        # RRQR rank corrupted to an in-range-but-wrong value.
        rc = BFLARRQRCache(Native)
        prepare!(rc, 3, p)
        factorize!(rc, A)
        rc.rank = 0
        @test_throws ArgumentError solve(rc, b)
    end

    @testset "wrong factor shape rejected" begin
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        cache.factors = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        @test_throws DimensionMismatch solve(cache, b)
    end

    @testset "NaN factor rejected" begin
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        cache.factors[1, 1] = mk(NaN)
        @test_throws DomainError solve(cache, b)
    end

    @testset "NaN RHS rejected" begin
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        bad = [mk(1), mk(NaN), mk(3)]
        @test_throws DomainError solve(cache, bad)
    end

    @testset "allocating solve does not bypass checked contract" begin
        # A corrupted factor must be rejected even though the destination is
        # freshly owned (the old solve_trusted!-based path would have skipped
        # the factor rescan).
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, A)
        cache.pivots[2] = 1  # in-range but inconsistent with perm
        @test_throws ArgumentError solve(cache, b)
    end
end
