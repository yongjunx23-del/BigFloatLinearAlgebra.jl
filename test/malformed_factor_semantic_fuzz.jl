# Malformed-factor *semantic* fuzz (v0.2.1).
#
# Corrupt a valid factor with in-range-but-wrong values (rank, threshold,
# reference_scale, tolerance-vs-atol, perm/pivots inconsistency, blocks/
# subdiag inconsistency, tolerance precision mismatch) and assert the checked API
# either throws a known exception or succeeds *and* passes a residual/backward
# error gate. A semantic corruption must never silently produce a wrong result.
# Run under both normal and --check-bounds=yes.

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

function _semantic_square(n, p, seed)
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

function _semantic_indef(n, p, seed)
    rng = MersenneTwister(seed)
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in j:n
        v = i == j ? BigFloat(isodd(i) ? i + 2 : -(i + 2); precision = p) :
            BigFloat(rand(rng, -8:8) // 16; precision = p)
        A[i, j] = v
        A[j, i] = i == j ? v : BigFloat(v; precision = p)
    end
    return A
end

function _semantic_assert_safe_solve(solve_fn, A, x, b, p)
    try
        solve_fn()
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

@testset "malformed-factor semantic fuzz" begin
    p = 256
    n = 8
    rng = MersenneTwister(20260825)
    A = _semantic_square(n, p, 1)
    Aind = _semantic_indef(n, p, 2)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    x = owned_zeros(BigFloat, n; precision_bits = p)

    # --- LU: perm/pivots finite but inconsistent ---
    for _ in 1:200
        c = BFLALUCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        kind = rand(rng, 1:4)
        if kind == 1
            c.pivots[rand(rng, 1:n)] = rand(rng, 1:n)  # in-range but maybe < k
        elseif kind == 2
            c.perm = shuffle(rng, collect(1:n))  # valid perm but inconsistent
        elseif kind == 3
            c.pivots = shuffle(rng, collect(1:n))  # valid pivots but inconsistent
        elseif kind == 4
            c.perm[rand(rng, 1:n)] = rand(rng, 1:n)  # may break permutation
        end
        result = _semantic_assert_safe_solve(() -> solve!(x, c, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end

    # --- LDLT: blocks/subdiag finite but inconsistent ---
    for _ in 1:200
        c = BFLALDLTCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, Aind)
        kind = rand(rng, 1:4)
        if kind == 1
            c.blocks = [rand(rng, 1:3) for _ in 1:n]  # may contain 3 / wrong sum
        elseif kind == 2
            c.subdiag_is_d = falses(n)  # inconsistent with blocks
        elseif kind == 3
            c.blocks = [1, 1, 1, 1, 1, 1, 1, 2]  # sum 10 != 8
        elseif kind == 4
            resize!(c.subdiag_is_d, n - 1)  # wrong length
        end
        result = _semantic_assert_safe_solve(() -> solve!(x, c, b), Aind, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end

    # --- RRQR: in-range-but-wrong rank / threshold / scale / tolerance ---
    for _ in 1:300
        c = BFLARRQRCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        kind = rand(rng, 1:7)
        if kind == 1
            c.rank = rand(rng, 0:n)  # in range but may be wrong
        elseif kind == 2
            c.effective_threshold = BigFloat(rand(rng) / 4; precision = p)  # finite, wrong
        elseif kind == 3
            c.reference_scale = BigFloat(rand(rng); precision = p)  # finite, wrong
        elseif kind == 4
            c.tolerance = BigFloat(rand(rng) / 4; precision = p)  # != atol
        elseif kind == 5
            c.atol = BigFloat(0; precision = 128)  # precision mismatch
        elseif kind == 6
            c.rtol = BigFloat(0; precision = 128)  # precision mismatch
        elseif kind == 7
            c.effective_threshold = BigFloat(0; precision = 128)  # precision mismatch
        end
        result = _semantic_assert_safe_solve(() -> solve!(x, c, b), A, x, b, p)
        @test result in (:ok, :expected)
        @test result != :bad_residual
    end
end
