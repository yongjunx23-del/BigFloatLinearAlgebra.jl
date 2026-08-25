# Malformed-factor fuzz: corrupt a valid factor's metadata and assert the
# checked API only succeeds or throws a *known* exception (never a BoundsError,
# segfault, or silent wrong result). Run under both normal and --check-bounds=yes.

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

function _square(n, p, seed)
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

function _indef(n, p, seed)
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

# Run a checked operation and assert it either succeeds or throws an expected
# exception (never a crash / BoundsError / unexpected error).
function _assert_safe(f)
    try
        f()
        return :ok
    catch e
        e isa EXPECTED && return :expected
        # A BoundsError or any unexpected error is a failure.
        return e
    end
end

@testset "malformed-factor fuzz" begin
    p = 256
    n = 8
    rng = MersenneTwister(20260824)
    A = _square(n, p, 1)
    Aind = _indef(n, p, 2)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    x = owned_zeros(BigFloat, n; precision_bits = p)

    # --- LU metadata corruptions ---
    for _ in 1:200
        c = BFLALUCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        kind = rand(rng, 1:6)
        if kind == 1
            c.pivots[rand(rng, 1:n)] = rand(rng, 0:(n + 3))  # out of range
        elseif kind == 2
            c.pivots[rand(rng, 1:n)] = rand(rng, 1:(n - 1))  # pivot < k
        elseif kind == 3
            c.perm[rand(rng, 1:n)] = rand(rng, 1:n)  # may break permutation
        elseif kind == 4
            c.perm = shuffle(rng, collect(1:n))  # valid perm but maybe inconsistent
        elseif kind == 5
            resize!(c.pivots, n - 1)  # wrong length
        elseif kind == 6
            resize!(c.perm, n + 1)  # wrong length
        end
        @test _assert_safe(() -> solve!(x, c, b)) in (:ok, :expected)
        @test _assert_safe(() -> factor_diagnostics(c)) in (:ok, :expected)
    end

    # --- LDLT metadata corruptions ---
    for _ in 1:200
        c = BFLALDLTCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, Aind)
        kind = rand(rng, 1:6)
        if kind == 1
            c.perm[rand(rng, 1:n)] = rand(rng, 1:n)
        elseif kind == 2
            c.blocks = [rand(rng, 1:3) for _ in 1:n]  # may contain 3 / wrong sum
        elseif kind == 3
            c.blocks = [1, 1, 1, 1, 1, 1, 1, 2]  # sum 10 != 8
        elseif kind == 4
            resize!(c.subdiag_is_d, n - 1)
        elseif kind == 5
            c.subdiag_is_d = falses(n)  # inconsistent with blocks
        elseif kind == 6
            resize!(c.perm, n + 1)
        end
        @test _assert_safe(() -> solve!(x, c, b)) in (:ok, :expected)
        @test _assert_safe(() -> factor_diagnostics(c)) in (:ok, :expected)
    end

    # --- RRQR metadata corruptions ---
    for _ in 1:200
        c = BFLARRQRCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        kind = rand(rng, 1:7)
        if kind == 1
            resize!(c.tau, n - 1)  # wrong tau length
        elseif kind == 2
            c.tau[1] = BigFloat(0; precision = 128)  # wrong tau precision
        elseif kind == 3
            c.jpvt[rand(rng, 1:n)] = rand(rng, 0:(n + 2))  # out of range
        elseif kind == 4
            c.jpvt = shuffle(rng, collect(1:n))  # valid perm
        elseif kind == 5
            c.rank = rand(rng, -1:(n + 2))  # out of range
        elseif kind == 6
            c.atol = BigFloat(NaN; precision = p)  # non-finite rank scalar
        elseif kind == 7
            c.effective_threshold = BigFloat(-1; precision = p)  # negative
        end
        @test _assert_safe(() -> solve!(x, c, b)) in (:ok, :expected)
        @test _assert_safe(() -> factor_diagnostics(c)) in (:ok, :expected)
    end

    # --- Cholesky: NaN / precision corruption ---
    for _ in 1:100
        c = BFLACholeskyCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        if rand(rng) < 0.5
            factor_matrix(c)[1, 1] = BigFloat(NaN; precision = p)
        else
            factor_matrix(c)[1, 1] = BigFloat(1; precision = 128)
        end
        @test _assert_safe(() -> solve!(x, c, b)) in (:ok, :expected)
        @test _assert_safe(() -> factor_diagnostics(c)) in (:ok, :expected)
    end
end
