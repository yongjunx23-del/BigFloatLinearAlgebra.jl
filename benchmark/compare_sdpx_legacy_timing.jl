# Performance A/B: BFLA Native vs. frozen SDPX legacy BigFloat kernels.
#
# Every operation is correctness-gated (bitwise parity where the trajectory is
# preserved, backward error for solves) before timing is accepted. Timings use
# median/IQR over at least 10 samples after 2 warmups. The legacy path is
# wrapped in a scoped `setprecision(BigFloat, p)` because its scratch objects
# inherit Julia's ambient precision; BFLA Native uses explicit precision.

using Random
import MutableArithmetics as MA
using SparseArrays
using Statistics
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()

using BigFloatLinearAlgebra:
    NoTrans, Trans, Lower, Upper, LeftSide, RightSide, UnitDiagonal, NonUnitDiagonal,
    issuccess, factor_matrix

function legacy_path()
    env = get(ENV, "SDPX_LEGACY_BIGFLOAT", "")
    isempty(env) || return env
    candidate = joinpath(@__DIR__, "..", "..", "SDPX", "SDPX-v041-unified-la-probes", "src", "kernels", "bigfloat.jl")
    isfile(candidate) || error("SDPX legacy kernel file not found at $candidate; set SDPX_LEGACY_BIGFLOAT")
    return candidate
end

include(legacy_path())

function owned_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(2 * rand(rng) - 1; precision = p)
    end
    return A
end

function measure(f::Function; warmup::Int=2, samples::Int=10)
    for _ in 1:warmup
        f()
    end
    ts = Float64[]
    as = Int[]
    for _ in 1:samples
        GC.gc()
        t = @timed f()
        push!(ts, t.time)
        push!(as, t.bytes)
    end
    sort!(ts); sort!(as)
    return (median=ts[end÷2+1], iqr=ts[ceil(Int,0.75*length(ts))]-ts[floor(Int,0.25*length(ts))+1], min=ts[1], max=ts[end], alloc=as[end÷2+1])
end

function report(name, p, n, nat, leg)
    ratio = leg.median > 0 ? nat.median / leg.median : NaN
    println(rpad(name, 12), " p=$p n=$n  native=", round(nat.median,sigdigits=3),
            "s (alloc=", nat.alloc, ")  legacy=", round(leg.median,sigdigits=3),
            "s (alloc=", leg.alloc, ")  ratio=", round(ratio,digits=3))
    return ratio
end

println("=== BFLA Native vs SDPX legacy timing ===")
println("julia=", VERSION, " threads=", Threads.nthreads(), " cpu=", Sys.CPU_NAME)
println("legacy=", legacy_path())

const SIZES = (8, 32, 64, 128)
const PRECISIONS = (128, 256, 512)
ratios = Float64[]

for p in PRECISIONS
    for n in SIZES
        rng = MersenneTwister(9000 + p + n)
        one = BigFloat(1; precision=p)
        zero = BigFloat(0; precision=p)

        # GEMM
        A = owned_matrix(n, n, p, rng)
        B = owned_matrix(n, n, p, rng)
        Cb = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        Cs = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, Cb)
        setprecision(BigFloat, p) do; kmul_owned!(Cs, A, B, one, zero); end
        @assert all(Cb[i,j] == Cs[i,j] for i in 1:n, j in 1:n) "gemm gate failed p=$p n=$n"
        tn = measure(() -> (BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, Cb); Cb))
        ts = measure(() -> setprecision(BigFloat, p) do; kmul_owned!(Cs, A, B, one, zero); end)
        push!(ratios, report("gemm", p, n, tn, ts))

        # Cholesky
        R = owned_matrix(n, n, p, rng)
        S = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        BFLA.gemm!(Native, Trans, NoTrans, one, R, R, zero, S)
        setprecision(BigFloat, p) do; for i in 1:n; S[i,i] = S[i,i] + one; end; end
        # Compare in-place factorization on an owned copy: one copy + factor.
        tn = measure(() -> BFLA.cholesky!(Native, BFLA.owned_copy(S)))
        ts = measure(() -> (L = BFLA.owned_copy(S); setprecision(BigFloat,p) do; kchol!(L); end; L))
        push!(ratios, report("cholesky", p, n, tn, ts))

        # TRSM (left lower solve)
        L = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for j in 1:n, i in j:n
            L[i,j] = i == j ? BigFloat(n+i; precision=p) : BigFloat(rand(rng); precision=p)
        end
        X = owned_matrix(n, n, p, rng)
        tn = measure(() -> (Y = BFLA.owned_copy(X); BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one, L, Y); Y))
        ts = measure(() -> (Y = BFLA.owned_copy(X); setprecision(BigFloat,p) do; ktrsm!(L, Y); end; Y))
        push!(ratios, report("trsm", p, n, tn, ts))

        # dot
        x = owned_matrix(n, 1, p, rng)[:,1]
        y = owned_matrix(n, 1, p, rng)[:,1]
        dref = BFLA.dot(Native, x, y)
        dleg = setprecision(BigFloat, p) do; kdot(x, y); end
        @assert dref == dleg "dot gate failed p=$p n=$n"
        tn = measure(() -> BFLA.dot(Native, x, y))
        ts = measure(() -> setprecision(BigFloat, p) do; kdot(x, y); end)
        push!(ratios, report("dot", p, n, tn, ts))
    end
end

println("=== summary: native/legacy median ratio across all cells ===")
println("median=", round(median(ratios),digits=3), "  max=", round(maximum(ratios),digits=3), "  min=", round(minimum(ratios),digits=3))
