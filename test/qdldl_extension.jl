import QDLDL
using SparseArrays

function _bf_qdldl_upper(p::Int; shift="0", offdiag="0.25")
    values = BFLA.owned_zeros(BigFloat, 5; precision_bits=p)
    values[1] = BigFloat(2; precision=p) + BigFloat(shift; precision=p)
    values[2] = BigFloat(1; precision=p)
    values[3] = BigFloat(-2; precision=p)
    values[4] = BigFloat(offdiag; precision=p)
    values[5] = BigFloat("-1.5"; precision=p)
    return SparseMatrixCSC(3, 3, [1, 2, 4, 6], [1, 1, 2, 2, 3], values)
end

function _bf_qdldl_tiny_upper(p::Int)
    values = BFLA.owned_zeros(BigFloat, 3; precision_bits=p)
    values[1] = BigFloat("1e-20"; precision=p)
    values[2] = BigFloat(-2; precision=p)
    values[3] = BigFloat(-3; precision=p)
    return SparseMatrixCSC(3, 3, [1, 2, 3, 4], [1, 2, 3], values)
end

function _bf_qdldl_dense(A, p)
    K = BFLA.owned_zeros(BigFloat, size(A, 1), size(A, 2); precision_bits=p)
    for column in axes(A, 2), pointer in nzrange(A, column)
        row = A.rowval[pointer]
        K[row, column] = A.nzval[pointer]
        K[column, row] = A.nzval[pointer]
    end
    return K
end

@testset "optional QDLDL BigFloat sparse LDL cache" begin
    @test Threads.nthreads() == 1
    @test BFLA.sparse_ldlt_available(BigFloat)
    p = 256
    setprecision(BigFloat, p) do
        A = _bf_qdldl_upper(p)
        signs = Int[1, -1, -1]
        cache = BFLA.sparse_ldlt_cache(
            A; precision_bits=p, dsigns=signs, nrhs=2,
        )
        signs[1] = -1
        @test cache.dsigns == Int[1, -1, -1]
        @test something(cache.factor).workspace.Dsigns === nothing
        @test iszero(something(cache.factor).workspace.regularize_eps)
        @test iszero(something(cache.factor).workspace.regularize_delta)
        @test BFLA.factor_status(cache).kind === :unprepared
        @test all(cache.matrix.nzval[i] !== A.nzval[j]
                  for i in eachindex(cache.matrix.nzval), j in eachindex(A.nzval))
        BFLA.factorize!(cache, A)
        @test BFLA.issuccess(cache)
        diagnostics = BFLA.factor_diagnostics(cache)
        @test diagnostics.provider === :qdldl
        @test diagnostics.symbolic_count == 1
        @test diagnostics.numeric_factor_count == 1
        @test diagnostics.positive_inertia == 1
        @test diagnostics.regularized_entries == 0
        @test diagnostics.precision_bits == p
        factor = something(cache.factor)
        @test all(precision(value) == p for storage in (
            factor.L.nzval, factor.Dinv.diag,
        ) for value in storage)

        # QDLDL's hidden triuA must own values independently of input/cache.
        @test all(hidden !== public for hidden in factor.workspace.triuA.nzval
                  for public in cache.matrix.nzval)
        @test all(hidden !== input for hidden in factor.workspace.triuA.nzval
                  for input in A.nzval)

        # Public factor_matrix is a detached ownership-safe snapshot.
        exposed = BFLA.factor_matrix(cache)
        exposed.colptr[end] -= 1
        exposed.nzval[1] += BigFloat(1; precision=p)
        @test cache.matrix.colptr == cache.frozen_colptr
        @test cache.matrix.nzval == cache.factored_values

        b = BFLA.owned_zeros(BigFloat, 3; precision_bits=p)
        b[1] = 1; b[2] = 2; b[3] = -1
        x = BFLA.owned_zeros(BigFloat, 3; precision_bits=p)
        BFLA.solve_trusted!(x, cache, b)
        @test all(x[i] !== b[j] for i in eachindex(x), j in eachindex(b))
        residual = LinearAlgebra.norm(_bf_qdldl_dense(A, p) * x - b, Inf)
        @test residual <= BigFloat("1e-70"; precision=p)

        A2 = _bf_qdldl_upper(p; shift="0.125", offdiag="0")
        BFLA.factorize!(cache, A2)
        @test BFLA.factor_diagnostics(cache).symbolic_count == 1
        @test BFLA.factor_diagnostics(cache).numeric_factor_count == 2

        B = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits=p)
        BFLA.copy_owned!(@view(B[:, 1]), b)
        B[1, 2] = 2; B[2, 2] = -1; B[3, 2] = 1
        X = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits=p)
        BFLA.solve_trusted!(X, cache, B)
        @test LinearAlgebra.norm(_bf_qdldl_dense(A2, p) * X - B, Inf) <=
              BigFloat("1e-70"; precision=p)
        @test BFLA.factor_diagnostics(cache).solve_count == 3
        @test_throws ArgumentError BFLA.solve_trusted!(B, cache, B)

        # Destination aliases against factor storage are rejected.
        factor = something(cache.factor)
        @test_throws ArgumentError BFLA.solve_trusted!(factor.Dinv.diag, cache, b)

        # Wrong precision/nonfinite RHS fail before destination mutation.
        rhs64 = BFLA.owned_zeros(BigFloat, 3; precision_bits=64)
        dest_before = BFLA.owned_copy(x; precision_bits=p)
        @test_throws BFLA.PrecisionMismatch BFLA.solve!(x, cache, rhs64)
        @test x == dest_before
        nonfinite = BFLA.owned_copy(b; precision_bits=p)
        nonfinite[2] = BigFloat(NaN; precision=p)
        @test_throws DomainError BFLA.solve_trusted!(x, cache, nonfinite)
        @test x == dest_before

        # Numeric mutation after factorization invalidates solve authority.
        cache.matrix.nzval[1] += BigFloat(1; precision=p)
        @test_throws ArgumentError BFLA.solve_trusted!(x, cache, b)
        @test BFLA.factor_status(cache).kind === :unprepared
        BFLA.factorize!(cache, A2)

        # Internal structural mutation is detected independently of input.
        saved_row = cache.matrix.rowval[2]
        cache.matrix.rowval[2] = saved_row == 1 ? 2 : 1
        @test_throws ArgumentError BFLA.factorize!(cache, A2)
        @test BFLA.factor_status(cache).kind === :unprepared
        cache.matrix.rowval[2] = saved_row
        BFLA.factorize!(cache, A2)

        drift = SparseMatrixCSC(
            3, 3, [1, 2, 4, 5], copy(A2.rowval[1:4]),
            BFLA.owned_copy(A2.nzval[1:4]; precision_bits=p),
        )
        @test_throws ArgumentError BFLA.factorize!(cache, drift)
        @test BFLA.factor_status(cache).kind === :unprepared

        # Nonfinite/ambient preflight attempts revoke prior solve authority.
        BFLA.factorize!(cache, A2)
        bad_nonfinite = _bf_qdldl_upper(p)
        bad_nonfinite.nzval[1] = BigFloat(NaN; precision=p)
        BFLA.factorize!(cache, bad_nonfinite; check=false)
        @test BFLA.factor_status(cache).kind === :unprepared
        @test BFLA.factor_diagnostics(cache).positive_inertia == -1
        @test_throws ArgumentError BFLA.solve_trusted!(x, cache, b)
        BFLA.factorize!(cache, A2)
        setprecision(BigFloat, 64) do
            @test_throws BFLA.PrecisionMismatch BFLA.factorize!(cache, A2)
        end
        @test BFLA.factor_status(cache).kind === :unprepared
        @test BFLA.factor_diagnostics(cache).positive_inertia == -1

        # Actual zero-pivot failure clears stale diagnostics and solve authority.
        BFLA.factorize!(cache, A2)
        bad = SparseMatrixCSC(
            3, 3, copy(A2.colptr), copy(A2.rowval),
            BFLA.owned_zeros(BigFloat, length(A2.nzval); precision_bits=p),
        )
        BFLA.factorize!(cache, bad; check=false)
        @test !BFLA.issuccess(cache)
        failed_diag = BFLA.factor_diagnostics(cache)
        @test failed_diag.positive_inertia == -1
        @test failed_diag.regularized_entries == 0
        @test_throws ArgumentError BFLA.solve_trusted!(x, cache, b)
        BFLA.factorize!(cache, A)

        # Tiny legitimate pivots are not absolutely regularized.
        tiny = _bf_qdldl_tiny_upper(p)
        tiny_cache = BFLA.sparse_ldlt_cache(
            tiny; precision_bits=p, dsigns=Int[1, -1, -1], nrhs=1,
        )
        BFLA.factorize!(tiny_cache, tiny)
        tiny_b = BFLA.owned_zeros(BigFloat, 3; precision_bits=p)
        BFLA.copy_owned!(tiny_b, tiny.nzval)
        tiny_x = BFLA.owned_zeros(BigFloat, 3; precision_bits=p)
        BFLA.solve_trusted!(tiny_x, tiny_cache, tiny_b)
        tiny_residual = LinearAlgebra.norm(
            _bf_qdldl_dense(tiny, p) * tiny_x - tiny_b, Inf,
        )
        @test tiny_residual <= BigFloat("1e-70"; precision=p)
        @test BFLA.factor_diagnostics(tiny_cache).regularized_entries == 0
        @test all(precision(value) == p for value in tiny_x)
    end
end
