# Metadata accessor / diagnostics consistency (v0.2.1).
#
# Accessors that read factor internals (`factor_perm`, `factor_pivots`,
# `factor_blocks`, `factor_inertia`, `factor_rank`, `factor_jpvt`,
# `factor_Rdiag`, `factor_rank_threshold`, `factor_diagnostics`,
# `numerical_rank`) must validate the factor before returning a value, so a
# caller cannot get a plausible-looking value from a factor whose metadata was
# externally mutated. Status/backend/precision/kind accessors need not validate.

using BigFloatLinearAlgebra

@testset "factor accessor integrity" begin
    p = 128
    mk(x) = BigFloat(x; precision = p)

    A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
    for i in 1:3, j in 1:3
        A[i, j] = mk(i == j ? 4.0 : 1.0)
    end

    @testset "status/backend/precision/kind need no validation" begin
        # These read only the factor's recorded metadata, not mutable internals.
        F = BFLA.lu(Native, A)
        @test factor_status(F).kind === :success
        @test factor_backend(F) isa BFLA.NativeBackend
        @test factor_precision(F) == p
        @test factor_kind(F) === :lu
    end

    @testset "LU accessors validate" begin
        F = BFLA.lu(Native, A)
        @test factor_perm(F) == collect(1:3)
        @test factor_pivots(F) == collect(1:3)
        # Corrupt the pivots; the accessor must reject.
        F.pivots[1] = 5
        @test_throws ArgumentError factor_perm(F)
        @test_throws ArgumentError factor_pivots(F)
        @test_throws ArgumentError factor_diagnostics(F)
    end

    @testset "LDLT accessors validate" begin
        F = BFLA.ldlt(Native, A)
        @test factor_perm(F) == collect(1:3)
        @test factor_blocks(F) == [1, 1, 1]
        @test factor_inertia(F) == (3, 0, 0)
        # Corruption is exercised on the mutable cache (the ordinary factor is
        # immutable); the accessor must reject a corrupted cache.
        dc = BFLALDLTCache(Native)
        prepare!(dc, 3, p)
        factorize!(dc, A)
        dc.blocks = [1, 1, 1, 1]
        @test_throws ArgumentError factor_perm(dc)
        @test_throws ArgumentError factor_blocks(dc)
        @test_throws ArgumentError factor_inertia(dc)
        @test_throws ArgumentError factor_diagnostics(dc)
    end

    @testset "RRQR accessors validate" begin
        F = BFLA.qr(Native, A)
        @test factor_rank(F) == 3
        @test factor_jpvt(F) == collect(1:3)
        @test length(factor_Rdiag(F)) == 3
        @test factor_rank_threshold(F) >= 0
        @test numerical_rank(F) == 3
        # Corruption is exercised on the mutable cache.
        rc = BFLARRQRCache(Native)
        prepare!(rc, 3, p)
        factorize!(rc, A)
        rc.rank = 0
        @test_throws ArgumentError factor_rank(rc)
        @test_throws ArgumentError factor_jpvt(rc)
        @test_throws ArgumentError factor_Rdiag(rc)
        @test_throws ArgumentError factor_rank_threshold(rc)
        @test_throws ArgumentError factor_diagnostics(rc)
    end

    @testset "cache accessors validate" begin
        lc = BFLALUCache(Native)
        prepare!(lc, 3, p)
        factorize!(lc, A)
        @test factor_perm(lc) == collect(1:3)
        @test factor_pivots(lc) == collect(1:3)
        lc.pivots[1] = 5
        @test_throws ArgumentError factor_perm(lc)
        @test_throws ArgumentError factor_pivots(lc)

        dc = BFLALDLTCache(Native)
        prepare!(dc, 3, p)
        factorize!(dc, A)
        @test factor_perm(dc) == collect(1:3)
        @test factor_blocks(dc) == [1, 1, 1]
        dc.blocks = [1, 1, 1, 1]
        @test_throws ArgumentError factor_perm(dc)
        @test_throws ArgumentError factor_blocks(dc)

        rc = BFLARRQRCache(Native)
        prepare!(rc, 3, p)
        factorize!(rc, A)
        @test factor_rank(rc) == 3
        @test factor_jpvt(rc) == collect(1:3)
        @test length(factor_Rdiag(rc)) == 3
        @test factor_rank_threshold(rc) >= 0
        rc.rank = 0
        @test_throws ArgumentError factor_rank(rc)
        @test_throws ArgumentError factor_jpvt(rc)
        @test_throws ArgumentError factor_Rdiag(rc)
        @test_throws ArgumentError factor_rank_threshold(rc)
    end
end
