# Cache `prepare!` lifecycle and parameter consistency (v0.2.1).
#
# The unified `_validate_cache_prepare` preflight runs before any cache field is
# mutated, so a failed `prepare!` leaves the cache in its previous state, and a
# legal re-`prepare!` succeeds.

using BigFloatLinearAlgebra

@testset "cache prepare contract" begin
    p = 128

    @testset "invalid parameters rejected" begin
        for (make_cache, name) in (
            (() -> BFLACholeskyCache(Native), "Cholesky"),
            (() -> BFLALUCache(Native), "LU"),
            (() -> BFLALDLTCache(Native), "LDLT"),
            (() -> BFLARRQRCache(Native), "RRQR"),
        )
            @testset "$name" begin
                cache = make_cache()
                @test_throws ArgumentError prepare!(cache, 3, p; nrhs = 0)
                @test_throws ArgumentError prepare!(cache, 3, p; nrhs = -1)
                @test_throws ArgumentError prepare!(cache, 3, p; workspace_workers = 0)
                @test_throws ArgumentError prepare!(cache, 3, p; workspace_workers = -1)
                @test_throws ArgumentError prepare!(cache, 3, 0)
                @test_throws ArgumentError prepare!(cache, 3, -1)
                @test_throws ArgumentError prepare!(cache, -1, p)
                # None of the failed calls may have prepared the cache.
                @test !factor_prepared(cache)
            end
        end
    end

    @testset "failed prepare leaves cache unchanged" begin
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        factorize!(cache, BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p))
        @test factor_prepared(cache)
        @test factor_size(cache) == 3
        # A failed re-prepare must not disturb the existing prepared state.
        @test_throws ArgumentError prepare!(cache, 3, p; nrhs = 0)
        @test factor_prepared(cache)
        @test factor_size(cache) == 3
    end

    @testset "legal reprepare succeeds" begin
        cache = BFLALUCache(Native)
        prepare!(cache, 3, p)
        @test factor_size(cache) == 3
        # Re-prepare at a different size/precision is an explicit act.
        prepare!(cache, 4, 256)
        @test factor_size(cache) == 4
        @test factor_precision(cache) == 256
        @test factor_prepared(cache)
    end
end
