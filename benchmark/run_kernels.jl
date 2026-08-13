include("bench_utils.jl")

# Benchmark configuration ---------------------------------------------------
const PRECISIONS = (128, 256, 512)
const SIZES = (1, 2, 8, 32, 64, 128, 256)
const SAMPLES = 10
const WARMUP = 2

println("=== BFLA benchmark ===")
println(environment())

function backward_error(A, x, b, p)
    n = size(A, 1)
    r = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
    BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), A, x, BigFloat(0; precision = p), r)
    setprecision(BigFloat, p) do
        for i in eachindex(r)
            r[i] = r[i] - b[i]
        end
    end
    return Float64(BFLA.norminf(Native, r))
end

function run_gemm(p::Int, n::Int)
    rng = MersenneTwister(1000 + p + n)
    A = owned_matrix(n, n, p, rng)
    B = owned_matrix(n, n, p, rng)
    C = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    one = BigFloat(1; precision = p)
    zero = BigFloat(0; precision = p)
    # correctness gate
    Cref = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.gemm!(Generic, NoTrans, NoTrans, one, A, B, zero, Cref)
    BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, C)
    @assert all(C[i, j] == Cref[i, j] for i in 1:n, j in 1:n) "gemm gate failed"
    tn = measure(() -> (BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, C); C); warmup = WARMUP, samples = SAMPLES)
    tg = measure(() -> (BFLA.gemm!(Generic, NoTrans, NoTrans, one, A, B, zero, C); C); warmup = WARMUP, samples = SAMPLES)
    println("gemm  p=$p n=$n  native=$(round(tn.median, sigdigits=3))s alloc=$(tn.allocs)  generic=$(round(tg.median, sigdigits=3))s alloc=$(tg.allocs)")
end

function run_cholesky(p::Int, n::Int)
    rng = MersenneTwister(2000 + p + n)
    R = owned_matrix(n, n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    one = BigFloat(1; precision = p)
    BFLA.gemm!(Native, Trans, NoTrans, one, R, R, BigFloat(0; precision = p), A)
    setprecision(BigFloat, p) do
        for i in 1:n
            A[i, i] = A[i, i] + one
        end
    end
    b = owned_matrix(n, 1, p, rng)
    # correctness gate
    F = BFLA.cholesky(Native, BFLA.owned_copy(A))
    x = BFLA.owned_copy(b)
    BFLA.solve!(F, x)
    @assert backward_error(A, x, vec(b), p) < 1e-20 "cholesky gate failed"
    tc = measure(() -> BFLA.cholesky(Native, BFLA.owned_copy(A)); warmup = WARMUP, samples = SAMPLES)
    ts = measure(() -> (y = BFLA.owned_copy(b); BFLA.solve!(F, y); y); warmup = WARMUP, samples = SAMPLES)
    println("chol p=$p n=$n  factor=$(round(tc.median, sigdigits=3))s alloc=$(tc.allocs)  solve=$(round(ts.median, sigdigits=3))s alloc=$(ts.allocs)")
end

function run_trsm(p::Int, n::Int)
    rng = MersenneTwister(3000 + p + n)
    L = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in j:n
        L[i, j] = i == j ? BigFloat(n + i; precision = p) : BigFloat(rand(rng); precision = p)
    end
    B = owned_matrix(n, n, p, rng)
    one = BigFloat(1; precision = p)
    Bn = BFLA.owned_copy(B)
    BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, L, Bn)
    Bg = BFLA.owned_copy(B)
    BFLA.trsm!(Generic, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, L, Bg)
    @assert all(Bn[i, j] == Bg[i, j] for i in 1:n, j in 1:n) "trsm gate failed"
    tn = measure(() -> (X = BFLA.owned_copy(B); BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, L, X); X); warmup = WARMUP, samples = SAMPLES)
    println("trsm p=$p n=$n  native=$(round(tn.median, sigdigits=3))s alloc=$(tn.allocs)")
end

for p in PRECISIONS
    for n in SIZES
        run_gemm(p, n)
        run_trsm(p, n)
        run_cholesky(p, n)
    end
end

println("=== done ===")
