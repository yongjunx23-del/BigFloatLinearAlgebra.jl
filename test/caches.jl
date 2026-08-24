# Reusable factor-cache tests: lifecycle, owned storage, precision contracts,
# zero-allocation hot paths, and Native/Generic numerical cross-checks.

using BigFloatLinearAlgebra
import MutableArithmetics as MA

const CACHES = (
    (BFLACholeskyCache, :cholesky),
    (BFLALUCache, :lu),
    (BFLALDLTCache, :ldlt),
    (BFLARRQRCache, :rrqr),
)

# SPD, indefinite, and square dense fixtures staged from exact rationals.
function _spd_fixture(n, p, rng)
    S = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        S[i, j] = BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    syrk!(
        NativeBackend(), Lower, NoTrans,
        BigFloat(1; precision = p), S, BigFloat(0; precision = p), A,
    )
    mirror_triangle!(A, Lower)
    for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(1; precision = p))
    end
    return A
end

function _indefinite_fixture(n, p, rng)
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in j:n
        v = if i == j
            BigFloat(isodd(i) ? i + 2 : -(i + 2); precision = p)
        else
            BigFloat(rand(rng, -8:8) // 16; precision = p)
        end
        A[i, j] = v
        A[j, i] = i == j ? v : BigFloat(v; precision = p)
    end
    return A
end

function _square_fixture(n, p, rng)
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        A[i, j] = BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(n + i; precision = p))
    end
    return A
end

function _rhs(A, x, p)
    b = owned_zeros(BigFloat, size(x)...; precision_bits = p)
    if x isa AbstractVector
        gemv!(
            NativeBackend(), NoTrans, BigFloat(1; precision = p), A, x,
            BigFloat(0; precision = p), b,
        )
    else
        gemm!(
            NativeBackend(), NoTrans, NoTrans, BigFloat(1; precision = p), A, x,
            BigFloat(0; precision = p), b,
        )
    end
    return b
end

function _backward_error(A, x, b, p)
    residual = owned_zeros(BigFloat, size(b)...; precision_bits = p)
    residual!(NativeBackend(), NoTrans, A, x, b, residual)
    return normwise_backward_error(NativeBackend(), NoTrans, A, x, b, residual)
end


# Local cross-backend closeness check at explicit precision.
function _cross_close(A, B, p; dimension::Int = length(A))
    maximum_value = BigFloat(0; precision = p)
    difference = BigFloat(0; precision = p)
    @inbounds for index in eachindex(A, B)
        MA.operate_to!(difference, -, A[index], B[index])
        signbit(difference) && MA.operate!(-, difference)
        difference > maximum_value &&
            MA.operate_to!(maximum_value, copy, difference)
    end
    scale = BigFloat(1; precision = p)
    for value in A
        abs(value) > scale && MA.operate_to!(scale, abs, value)
    end
    bound = BigFloat(100 * max(dimension, 1); precision = p) * _eps_p(p) * scale
    return maximum_value <= bound
end


# Unit roundoff at the explicit working precision (never ambient).
_eps_p(p) = BigFloat(2; precision = p) ^ (1 - p)

@testset "factor cache lifecycle and ownership" begin
    for p in (128, 256), n in (4, 16)
        rng = MersenneTwister(31 + p + n)
        A = _square_fixture(n, p, rng)
        x_true = vec(owned_zeros(BigFloat, n; precision_bits = p))
        for i in 1:n
            x_true[i] = BigFloat(rand(rng, -1024:1024); precision = p)
        end
        b = _rhs(A, x_true, p)

        c = BFLALUCache(NativeBackend())
        @test factor_prepared(c) == false
        @test factor_size(c) === nothing
        @test factor_status(c).kind == :unprepared
        # use before prepare fails explicitly
        @test_throws ArgumentError factorize!(c, A)

        prepare!(c, n, p)
        @test factor_prepared(c)
        @test factor_size(c) == n
        @test factor_precision(c) == p

        x = owned_zeros(BigFloat, n; precision_bits = p)
        @test_throws ArgumentError solve!(x, c, b)  # not factorized yet

        factorize!(c, A)
        @test issuccess(c)
        @test factor_status(c).kind == :success
        @test factor_kind(c) == :lu

        # repeated solve into an existing destination preserves identity
        solve!(x, c, b)
        id_before = objectid.(x)
        solve!(x, c, b)
        @test objectid.(x) == id_before
        @test _backward_error(A, x, b, p) <= BigFloat(100; precision = p) * _eps_p(p)

        # invalidate then refactor into the same owned storage
        invalidate!(c)
        @test factor_status(c).kind == :unprepared
        factorize!(c, A)
        @test issuccess(c)

        # multi-RHS solve
        X = owned_zeros(BigFloat, n, 3; precision_bits = p)
        Xtrue = owned_zeros(BigFloat, n, 3; precision_bits = p)
        for j in 1:3, i in 1:n
            Xtrue[i, j] = BigFloat(rand(rng, -1024:1024); precision = p)
        end
        Bm = _rhs(A, Xtrue, p)
        solve!(X, c, Bm)
        @test _backward_error(A, X, Bm, p) <= BigFloat(300; precision = p) * _eps_p(p)
    end
end

@testset "cache precision mismatch is explicit" begin
    p = 128
    n = 4
    rng = MersenneTwister(42)
    A = _square_fixture(n, p, rng)
    c = BFLALUCache(NativeBackend())
    prepare!(c, n, 256)
    # coefficient matrix at a different precision than the cache
    @test_throws PrecisionMismatch factorize!(c, A)
    # factorize at the cache precision, then a mismatched RHS
    A256 = owned_copy(A; precision_bits = 256)
    factorize!(c, A256)
    b128 = owned_zeros(BigFloat, n; precision_bits = 128)
    x = owned_zeros(BigFloat, n; precision_bits = 256)
    @test_throws PrecisionMismatch solve!(x, c, b128)
end

@testset "all four caches factor and solve at machine precision" begin
    for p in (128, 256)
        for (ctor, kind) in (
            (BFLACholeskyCache, :cholesky),
            (BFLALUCache, :lu),
            (BFLALDLTCache, :ldlt),
            (BFLARRQRCache, :rrqr),
        )
            n = 16
            rng = MersenneTwister(100p)
            A = kind == :cholesky ? _spd_fixture(n, p, rng) :
                kind == :ldlt ? _indefinite_fixture(n, p, rng) :
                _square_fixture(n, p, rng)
            x_true = owned_zeros(BigFloat, n; precision_bits = p)
            for i in 1:n
                x_true[i] = BigFloat(rand(rng, -1024:1024); precision = p)
            end
            b = _rhs(A, x_true, p)
            c = ctor(NativeBackend())
            prepare!(c, n, p)
            factorize!(c, A)
            @test issuccess(c)
            @test factor_kind(c) == kind
            x = owned_zeros(BigFloat, n; precision_bits = p)
            solve!(x, c, b)
            @test _backward_error(A, x, b, p) <= BigFloat(100; precision = p) * _eps_p(p)
        end
    end
end

@testset "cache backend is explicit and never falls back" begin
    c = BFLALUCache(NativeBackend())
    @test factor_backend(c) === NativeBackend()
    c2 = BFLALUCache(GenericBackend())
    @test factor_backend(c2) === GenericBackend()
end

@testset "cache refinement performs exactly one step" begin
    p = 256
    n = 8
    rng = MersenneTwister(7)
    A = _spd_fixture(n, p, rng)
    x0 = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        x0[i] = BigFloat(rand(rng, -1024:1024); precision = p)
    end
    b = _rhs(A, x0, p)
    c = BFLACholeskyCache(NativeBackend())
    prepare!(c, n, p)
    factorize!(c, A)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve!(x, c, b)
    report = refine_once!(c, A, x, b)
    @test haskey(report, :backward_error_after)
    @test isfinite(report.backward_error_after)
end

@testset "cache zero-allocation hot path" begin
    p = 256
    n = 32
    rng = MersenneTwister(9)
    A = _square_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    x = owned_zeros(BigFloat, n; precision_bits = p)

    # LU: repeated factorize + solve must not allocate after warm-up.
    c = BFLALUCache(NativeBackend())
    prepare!(c, n, p)
    factorize!(c, A)
    for _ in 1:20
        factorize!(c, A)
    end
    solve!(x, c, b)
    for _ in 1:5
        solve!(x, c, b)
    end
    @test @allocated(solve!(x, c, b)) == 0
    @test @allocated(factorize!(c, A)) == 0
    @test @allocated(begin factorize!(c, A); solve!(x, c, b); end) == 0

    # Cholesky: same contract.
    Aspd = _spd_fixture(n, p, rng)
    cc = BFLACholeskyCache(NativeBackend())
    prepare!(cc, n, p)
    factorize!(cc, Aspd)
    for _ in 1:5
        factorize!(cc, Aspd)
    end
    solve!(x, cc, b)
    for _ in 1:5
        solve!(x, cc, b)
    end
    @test @allocated(solve!(x, cc, b)) == 0
    @test @allocated(factorize!(cc, Aspd)) == 0
end

@testset "cache Native/Generic numerical cross-check" begin
    for (p, n) in ((128, 8), (256, 16))
        rng = MersenneTwister(17 + p)
        for (ctor, kind) in (
            (BFLACholeskyCache, :cholesky),
            (BFLALUCache, :lu),
            (BFLALDLTCache, :ldlt),
            (BFLARRQRCache, :rrqr),
        )
            A = kind == :cholesky ? _spd_fixture(n, p, rng) :
                kind == :ldlt ? _indefinite_fixture(n, p, rng) :
                _square_fixture(n, p, rng)
            x_true = owned_zeros(BigFloat, n; precision_bits = p)
            for i in 1:n
                x_true[i] = BigFloat(rand(rng, -1024:1024); precision = p)
            end
            b = _rhs(A, x_true, p)

            cn = ctor(NativeBackend())
            prepare!(cn, n, p)
            factorize!(cn, A)
            @test issuccess(cn)
            x_n = owned_zeros(BigFloat, n; precision_bits = p)
            solve!(x_n, cn, b)

            cg = ctor(GenericBackend())
            prepare!(cg, n, p)
            factorize!(cg, A)
            x_g = owned_zeros(BigFloat, n; precision_bits = p)
            solve!(x_g, cg, b)
            @test _cross_close(x_n, x_g, p; dimension = n)
            @test _backward_error(A, x_n, b, p) <= BigFloat(100; precision = p) * _eps_p(p)
        end
    end
end
