# Malformed-factor *shape* fuzz (v0.2.1).
#
# Randomly replace a valid factor's matrix shape and assert the checked API
# either throws a known exception or succeeds *and* passes a residual/backward
# error gate. A malformed shape must never reach an `@inbounds` kernel (no
# BoundsError, no segfault, no silent wrong result). Run under both normal and
# --check-bounds=yes.

using BigFloatLinearAlgebra
import MutableArithmetics as MA
using Random

const EXPECTED = Union{
    ArgumentError,
    DimensionMismatch,
    PrecisionMismatch,
    DomainError,
    LinearAlgebra.SingularException,
    LinearAlgebra.PosDefException,
}

function _shape_square(n, p, seed)
    rng = MersenneTwister(seed)
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        A[i, j] = BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(n + i; precision = p))
    end
    return A
end

# Run a checked solve and, if it succeeds, verify the backward error is small.
# Returns :ok (succeeded with a passing residual gate), :expected (threw a known
# exception), or the unexpected exception (a failure).
function _shape_assert_safe_solve(solve_fn, A, x, b, p)
    try
        solve_fn()
        # Success: the result must pass a residual/backward-error gate.
        r = owned_zeros(BigFloat, size(b)...; precision_bits = p)
        residual!(NativeBackend(), NoTrans, A, x, b, r)
        bound = max(100 * max(length(b), 1), 1) * BigFloat(2; precision = p)^(1 - p) *
            norminf(NativeBackend(), b)
        norminf(NativeBackend(), r) <= bound ? :ok : :bad_residual
    catch e
        e isa EXPECTED && return :expected
        return e
    end
end

@testset "malformed-factor shape fuzz" begin
    p = 256
    n = 8
    rng = MersenneTwister(20260825)
    A = _shape_square(n, p, 1)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    x = owned_zeros(BigFloat, n; precision_bits = p)

    # --- cache factor storage replaced with a wrong shape ---
    for _ in 1:200
        c = BFLALUCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        # Replace the factor storage with a wrong-size matrix.
        m = rand(rng, 1:(n + 2))
        k = rand(rng, 1:(n + 2))
        c.factors = owned_zeros(BigFloat, m, k; precision_bits = p)
        result = _shape_assert_safe_solve(() -> solve!(x, c, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end

    # --- cache.n inconsistent with the factor storage ---
    for _ in 1:100
        c = BFLALUCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        c.n = rand(rng, 1:(n + 2))
        result = _shape_assert_safe_solve(() -> solve!(x, c, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end

    # --- ordinary factors with a rectangular factor matrix ---
    for _ in 1:100
        m = rand(rng, 1:(n + 2))
        k = rand(rng, 1:(n + 2))
        rect = owned_zeros(BigFloat, m, k; precision_bits = p)
        F = BFLALUFactor(rect, NativeBackend(), p, BFLA.SUCCESS_STATUS, collect(1:k), collect(1:k))
        result = _shape_assert_safe_solve(() -> solve!(F, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end

    # --- RRQR cache rectangular factor storage (square-only cache) ---
    for _ in 1:100
        c = BFLARRQRCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        c.factors = owned_zeros(BigFloat, n + 1, n; precision_bits = p)
        result = _shape_assert_safe_solve(() -> solve!(x, c, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end
end
