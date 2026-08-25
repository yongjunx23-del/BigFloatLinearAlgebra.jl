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


# Diagonal-dominant square matrix that forces multiple row swaps under LU.
function _pivot_heavy_fixture(n, p)
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        A[i, j] = BigFloat((i + j) % 5 + 1; precision = p)
    end
    for i in 1:n
        A[i, i] = BigFloat(n - i + 5; precision = p)
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

        # repeated trusted solve into an existing owned destination preserves
        # object identity (the checked solve! re-owns by design)
        solve_trusted!(x, c, b)
        id_before = objectid.(x)
        solve_trusted!(x, c, b)
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
    prepare_refinement!(c, b)   # vector-RHS scratch shape
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
    solve_trusted!(x, c, b)
    for _ in 1:5
        solve_trusted!(x, c, b)
    end
    @test @allocated(solve_trusted!(x, c, b)) == 0
    @test @allocated(factorize!(c, A)) == 0
    @test @allocated(begin factorize!(c, A); solve_trusted!(x, c, b); end) == 0

    # Cholesky: same contract.
    Aspd = _spd_fixture(n, p, rng)
    cc = BFLACholeskyCache(NativeBackend())
    prepare!(cc, n, p)
    factorize!(cc, Aspd)
    for _ in 1:5
        factorize!(cc, Aspd)
    end
    solve_trusted!(x, cc, b)
    for _ in 1:5
        solve_trusted!(x, cc, b)
    end
    @test @allocated(solve_trusted!(x, cc, b)) == 0
    @test @allocated(factorize!(cc, Aspd)) == 0
    @test @allocated(begin factorize!(cc, Aspd); solve_trusted!(x, cc, b); end) == 0

    # LDLT and RRQR: trusted solve is zero-allocation; their factorize! may
    # allocate pivot/tau metadata (not gated to zero).
    Aind = _indefinite_fixture(n, p, rng)
    cl = BFLALDLTCache(NativeBackend())
    prepare!(cl, n, p)
    factorize!(cl, Aind)
    solve_trusted!(x, cl, b)
    for _ in 1:5
        solve_trusted!(x, cl, b)
    end
    @test @allocated(solve_trusted!(x, cl, b)) == 0

    cq = BFLARRQRCache(NativeBackend())
    prepare!(cq, n, p)
    factorize!(cq, A)
    solve_trusted!(x, cq, b)
    for _ in 1:5
        solve_trusted!(x, cq, b)
    end
    @test @allocated(solve_trusted!(x, cq, b)) == 0
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

@testset "checked solve! is ownership-safe on a shared destination" begin
    p = 256
    n = 8
    rng = MersenneTwister(99)
    A = _square_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    c = BFLALUCache(NativeBackend())
    prepare!(c, n, p)
    factorize!(c, A)
    # A shared-element destination (all slots point at one object) must not be
    # silently corrupted; the checked solve! re-owns it safely.
    x_shared = fill!(Array{BigFloat}(undef, n), BigFloat(0; precision = p))
    @test all(x_shared[i] === x_shared[1] for i in eachindex(x_shared))
    solve!(x_shared, c, b)
    @test is_independently_owned(x_shared)
    @test _backward_error(A, x_shared, b, p) <= BigFloat(100; precision = p) * _eps_p(p)
    # A shared destination with the wrong (ambient) precision is also repaired.
    x_ambient = fill!(Array{BigFloat}(undef, n), BigFloat(0))
    solve!(x_ambient, c, b)
    @test all(precision(value) == p for value in x_ambient)
end

@testset "LU cache Generic backend executes the reference path" begin
    p = 256
    n = 8
    rng = MersenneTwister(1234)
    A = _square_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i + 1; precision = p)
    end
    cn = BFLALUCache(NativeBackend())
    prepare!(cn, n, p)
    factorize!(cn, A)
    @test factor_backend(cn) === NativeBackend()
    x_n = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x_n, cn, b)

    cg = BFLALUCache(GenericBackend())
    prepare!(cg, n, p)
    factorize!(cg, A)
    @test factor_backend(cg) === GenericBackend()
    x_g = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x_g, cg, b)
    @test _cross_close(x_n, x_g, p; dimension = n)
    # the two backends dispatch to different kernels, so their recorded pivots
    # may differ; both must yield a valid, consistent factorization
    @test issuccess(cn) && issuccess(cg)
end

@testset "LU cache final permutation matches allocating LU on pivot-heavy input" begin
    p = 256
    for n in (4, 8)
        A = _pivot_heavy_fixture(n, p)
        c = BFLALUCache(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        F = lu(NativeBackend(), A; check = false)
        @test factor_perm(c) == factor_perm(F)
        @test factor_diagnostics(c).permutation == factor_perm(F)
        @test any(k -> c.pivots[k] != k, eachindex(c.pivots))  # pivoting occurred
        b = owned_zeros(BigFloat, n; precision_bits = p)
        for i in 1:n
            b[i] = BigFloat(i + 1; precision = p)
        end
        x = owned_zeros(BigFloat, n; precision_bits = p)
        solve_trusted!(x, c, b)
        @test _backward_error(A, x, b, p) <= BigFloat(200; precision = p) * _eps_p(p)
    end
end

@testset "cache refinement lifecycle and allocation" begin
    p = 256
    n = 16
    rng = MersenneTwister(5)
    A = _spd_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(rand(rng, -1024:1024); precision = p)
    end
    c = BFLACholeskyCache(NativeBackend())
    prepare!(c, n, p; nrhs = 1)
    prepare_refinement!(c, b)   # vector-RHS scratch shape
    @test c.refine !== nothing   # prepare_refinement! ran eagerly
    factorize!(c, A)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x, c, b)
    before = _backward_error(A, x, b, p)
    r = refine_once!(c, A, x, b)
    @test isfinite(r.backward_error_after)
    # one refinement step must not worsen the backward error (or stay equal)
    @test r.backward_error_after <= before + BigFloat(10; precision = p) * _eps_p(p)
    # invalidate/refactor keeps the owned refinement scratch reusable
    invalidate!(c)
    factorize!(c, A)
    x2 = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x2, c, b)
    r2 = refine_once!(c, A, x2, b)
    @test isfinite(r2.backward_error_after)
end

@testset "solve_trusted! rejects aliasing against factor and RHS" begin
    p = 256
    n = 8
    rng = MersenneTwister(2024)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    for (ctor, kind) in (
        (BFLACholeskyCache, :cholesky),
        (BFLALUCache, :lu),
        (BFLALDLTCache, :ldlt),
        (BFLARRQRCache, :rrqr),
    )
        A = kind == :cholesky ? _spd_fixture(n, p, rng) :
            kind == :ldlt ? _indefinite_fixture(n, p, rng) :
            _square_fixture(n, p, rng)
        c = ctor(NativeBackend())
        prepare!(c, n, p)
        factorize!(c, A)
        @test issuccess(c)
        x = owned_zeros(BigFloat, n; precision_bits = p)
        # x aliases a column of the factor matrix -> must be rejected.
        x_alias = view(factor_matrix(c), :, 1)
        @test_throws ArgumentError solve_trusted!(x_alias, c, b)
        # x aliases the RHS -> must be rejected.
        @test_throws ArgumentError solve_trusted!(b, c, b)
        # a correct owned destination solves fine.
        solve_trusted!(x, c, b)
        @test isfinite(_backward_error(A, x, b, p))
    end
end

@testset "metadata accessors are invalid after invalidate!/failure" begin
    p = 256
    n = 8
    rng = MersenneTwister(77)
    A = _square_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    c = BFLALUCache(NativeBackend())
    prepare!(c, n, p)
    factorize!(c, A)
    @test issuccess(c)
    @test factor_perm(c) == factor_perm(c)
    @test haskey(factor_diagnostics(c), :permutation)
    # after invalidate!: metadata accessors must throw, not return stale values
    invalidate!(c)
    @test_throws ArgumentError factor_perm(c)
    @test_throws ArgumentError factor_diagnostics(c)
    # after a non-finite input, status is nonfinite and metadata accessors throw
    factorize!(c, A)
    Anan = owned_copy(A)
    Anan[1, 1] = BigFloat(NaN; precision = p)
    factorize!(c, Anan)
    @test !issuccess(c)
    @test factor_status(c).kind == :nonfinite
    @test_throws ArgumentError factor_perm(c)
    @test_throws ArgumentError factor_diagnostics(c)
end

@testset "refinement allocation contract is honest" begin
    # refine_once! is allocation-light, NOT zero-allocation: it still builds fresh
    # BigFloat constants/scratch via the generic residual!/normwise_backward_error
    # path. We assert a bound (not == 0) so a regression that balloons allocation
    # is caught while the honest contract is recorded.
    p = 256
    n = 16
    rng = MersenneTwister(31337)
    A = _spd_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(rand(rng, -1024:1024); precision = p)
    end
    c = BFLACholeskyCache(NativeBackend())
    prepare!(c, n, p; nrhs = 1)
    prepare_refinement!(c, b)
    factorize!(c, A)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x, c, b)
    for _ in 1:20
        refine_once!(c, A, x, b)
    end
    alloc = @allocated refine_once!(c, A, x, b)
    @test alloc >= 0
    @test alloc < 20000  # allocation-light bound; refinement is NOT zero-alloc
    # storage object identity is preserved across refine_once!
    storage_before = c.refine
    refine_once!(c, A, x, b)
    @test c.refine === storage_before
end

@testset "refine_once! and refine_once_trusted! reject alias and repair ownership" begin
    p = 256
    n = 8
    rng = MersenneTwister(4242)
    A = _spd_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(rand(rng, -1024:1024); precision = p)
    end
    c = BFLACholeskyCache(NativeBackend())
    prepare!(c, n, p; nrhs = 1)
    prepare_refinement!(c, b)
    factorize!(c, A)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x, c, b)

    # x aliases the factor matrix column -> both checked and trusted reject.
    x_alias = view(factor_matrix(c), :, 1)
    @test_throws ArgumentError refine_once!(c, A, x_alias, b)
    @test_throws ArgumentError refine_once_trusted!(c, A, x_alias, b)
    # x aliases the RHS -> both reject.
    @test_throws ArgumentError refine_once!(c, A, b, b)
    @test_throws ArgumentError refine_once_trusted!(c, A, b, b)
    # a shared-element x is safely repaired by the checked path and rejected by
    # the trusted path (which requires independent ownership).
    x_shared = fill!(Array{BigFloat}(undef, n), BigFloat(0; precision = p))
    report = refine_once!(c, A, x_shared, b)   # repaired safely
    @test is_independently_owned(x_shared)
    @test isfinite(report.backward_error_after)
    # trusted path on an owned, correct-precision x works.
    x2 = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x2, c, b)
    r2 = refine_once_trusted!(c, A, x2, b)
    @test isfinite(r2.backward_error_after)
    # scratch shape mismatch throws (no silent re-resize): prepare for a matrix,
    # then refine with the vector RHS.
    cm = BFLACholeskyCache(NativeBackend())
    prepare!(cm, n, p; nrhs = 2)
    prepare_refinement!(cm, owned_zeros(BigFloat, n, 2; precision_bits = p))
    factorize!(cm, A)
    xm = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(xm, cm, b)
    @test_throws DimensionMismatch refine_once!(cm, A, xm, b)  # vector vs n×2 scratch
end

@testset "README reusable-cache example runs" begin
    import MutableArithmetics as MA
    n, p = 32, 256
    A = owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        A[i, j] = BigFloat(i == j ? n - i + 8 : (i + j) % 5 + 1; precision = p)
    end
    for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(n; precision = p))
    end
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i + 1; precision = p)
    end
    cache = BFLACholeskyCache(NativeBackend())
    prepare!(cache, n, p; nrhs = 1)
    factorize!(cache, A)
    @test issuccess(cache)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x, cache, b)
    @test isfinite(_backward_error(A, x, b, p))
    x2 = owned_zeros(BigFloat, n; precision_bits = p)
    fill!(x2, BigFloat(0; precision = p))
    solve!(x2, cache, b)
    @test isfinite(_backward_error(A, x2, b, p))
    invalidate!(cache)
    @test factor_status(cache).kind == :unprepared
end

@testset "prepare_refinement! lifecycle contract" begin
    p = 256
    n = 8
    rng = MersenneTwister(818)
    A = _spd_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    # use before prepare -> clear error
    c = BFLACholeskyCache(NativeBackend())
    @test_throws ArgumentError prepare_refinement!(c, b)
    @test_throws ArgumentError prepare_refinement!(c, 1)
    prepare!(c, n, p)
    # now prepare_refinement! works with a Vector template
    prepare_refinement!(c, b)
    factorize!(c, A)
    x = owned_zeros(BigFloat, n; precision_bits = p)
    solve_trusted!(x, c, b)
    r = refine_once!(c, A, x, b)
    @test isfinite(r.backward_error_after)
    # precision mismatch -> clear error
    bp = owned_zeros(BigFloat, n; precision_bits = 128)
    @test_throws PrecisionMismatch prepare_refinement!(c, bp)
    # shape mismatch (rows) -> clear error
    bw = owned_zeros(BigFloat, n + 1; precision_bits = p)
    @test_throws DimensionMismatch prepare_refinement!(c, bw)
    # shape change requires explicit re-prepare; refine_once! with a mismatched
    # scratch shape throws rather than silently resizing.
    prepare_refinement!(c, owned_zeros(BigFloat, n, 2; precision_bits = p))
    @test_throws DimensionMismatch refine_once!(c, A, x, b)  # vector vs n×2 scratch
    prepare_refinement!(c, b)  # restore vector scratch
    r2 = refine_once!(c, A, x, b)
    @test isfinite(r2.backward_error_after)
end

@testset "checked factor-integrity contract" begin
    p = 256
    n = 8
    rng = MersenneTwister(909)
    A = _square_fixture(n, p, rng)
    b = owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        b[i] = BigFloat(i; precision = p)
    end
    x = owned_zeros(BigFloat, n; precision_bits = p)

    # LU: a mutated factor matrix (NaN) is rejected by checked solve!, while the
    # trusted path keeps its explicit caller contract (no rescan).
    c = BFLALUCache(NativeBackend())
    prepare!(c, n, p)
    factorize!(c, A)
    Fm = factor_matrix(c)
    Fm[1, 1] = BigFloat(NaN; precision = p)
    @test_throws DomainError solve!(x, c, b)
    @test_throws DomainError refine_once!(c, A, x, b)
    solve_trusted!(x, c, b)  # trusted does not rescan factor storage

    # a precision-mutated factor matrix is rejected by checked
    c2 = BFLALUCache(NativeBackend())
    prepare!(c2, n, p)
    factorize!(c2, A)
    factor_matrix(c2)[1, 1] = BigFloat(1; precision = 128)
    @test_throws PrecisionMismatch solve!(x, c2, b)

    # invalid LU pivot metadata is rejected by checked
    c3 = BFLALUCache(NativeBackend())
    prepare!(c3, n, p)
    factorize!(c3, A)
    c3.pivots[1] = n + 5
    @test_throws ArgumentError solve!(x, c3, b)

    # LDLT: invalid block metadata rejected by checked.
    Aind = _indefinite_fixture(n, p, rng)
    cl = BFLALDLTCache(NativeBackend())
    prepare!(cl, n, p)
    factorize!(cl, Aind)
    cl.blocks = [1, 1, 1, 1, 1, 1, 1, 2]  # sums to 10 != 8
    @test_throws ArgumentError solve!(x, cl, b)

    # RRQR: invalid jpvt / rank rejected by checked.
    cq = BFLARRQRCache(NativeBackend())
    prepare!(cq, n, p)
    factorize!(cq, A)
    cq.rank = n + 3
    @test_throws ArgumentError solve!(x, cq, b)
    cq2 = BFLARRQRCache(NativeBackend())
    prepare!(cq2, n, p)
    factorize!(cq2, A)
    cq2.jpvt[1] = -1
    @test_throws ArgumentError solve!(x, cq2, b)
end
