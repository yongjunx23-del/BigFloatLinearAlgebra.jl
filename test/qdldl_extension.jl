using QDLDL
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

function _bf_qdldl_dense(A, p)
    K = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits=p)
    for column in 1:3, pointer in nzrange(A, column)
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
        cache = BFLA.sparse_ldlt_cache(
            A; precision_bits=p, dsigns=Int[1, -1, -1], nrhs=2,
            regularize_eps=BigFloat("1e-70"; precision=p),
            regularize_delta=BigFloat("1e-60"; precision=p),
        )
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
        @test diagnostics.precision_bits == p
        factor = something(cache.factor)
        @test all(precision(value) == p for storage in (
            factor.L.nzval, factor.Dinv.diag,
        ) for value in storage)

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

        drift = SparseMatrixCSC(
            3, 3, [1, 2, 4, 5], copy(A2.rowval[1:4]),
            BFLA.owned_copy(A2.nzval[1:4]; precision_bits=p),
        )
        @test_throws ArgumentError BFLA.factorize!(cache, drift)
        @test BFLA.issuccess(cache) # preflight drift preserves the old factor

        old_count = BFLA.factor_diagnostics(cache).numeric_factor_count
        setprecision(BigFloat, 64) do
            @test_throws BFLA.PrecisionMismatch BFLA.factorize!(cache, A2)
        end
        @test BFLA.factor_diagnostics(cache).numeric_factor_count == old_count
        @test BFLA.issuccess(cache)

        # Actual refactor failure invalidates the previous factor.
        stale = BFLA.sparse_ldlt_cache(
            A; precision_bits=p, dsigns=Int[1, -1, -1], nrhs=1,
            regularize_eps=BigFloat(0; precision=p),
            regularize_delta=BigFloat(0; precision=p),
        )
        BFLA.factorize!(stale, A)
        bad = SparseMatrixCSC(
            3, 3, copy(A.colptr), copy(A.rowval),
            BFLA.owned_zeros(BigFloat, length(A.nzval); precision_bits=p),
        )
        BFLA.factorize!(stale, bad; check=false)
        @test !BFLA.issuccess(stale)
        @test_throws ArgumentError BFLA.solve_trusted!(x, stale, b)
        BFLA.factorize!(stale, A)
        BFLA.solve_trusted!(x, stale, b)
        @test all(precision(value) == p for value in x)
    end
end
