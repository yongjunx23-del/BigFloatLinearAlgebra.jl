# Measure the validation (full-array uniform-precision scan) overhead relative
# to the underlying kernel, for operations where the scan is asymptotically
# comparable to the work itself (dot/axpy!/trsv!) versus GEMM/SYRK where the
# O(n^2) scan is dominated by O(n^3) arithmetic. Correctness is mandatory; this
# benchmark only quantifies cost so repeated factor-solve patterns can be
# measured before any architectural change.

using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

using BigFloatLinearAlgebra:
    NoTrans, Trans, Lower, Upper, LeftSide, NonUnitDiagonal, issuccess, factor_matrix

function owned_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(2 * rand(rng) - 1; precision = p)
    end
    return A
end

function measure(f::Function; warmup::Int = 3, samples::Int = 20)
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
    sort!(ts)
    return (median = ts[end ÷ 2 + 1], alloc = as[end ÷ 2 + 1])
end

println("=== precision-scan overhead ===")
println("julia=", VERSION, " threads=", Threads.nthreads())

for p in (128, 256, 512)
    for n in (64, 256, 1024)
        rng = MersenneTwister(7000 + p + n)
        x = owned_matrix(n, 1, p, rng)[:, 1]
        y = owned_matrix(n, 1, p, rng)[:, 1]
        a = BigFloat(2; precision = p)

        # Public dot = validation scan + kernel; raw scan alone quantifies the
        # boundary cost. The kernel itself is measured via an already-validated
        # internal call so the difference is attributable to validation.
        pval = BFLA._check_precision(x, y)
        t_pub = measure(() -> BFLA.dot(Native, x, y))
        t_scan = measure(() -> BFLA._check_precision(x, y))
        t_kernel = measure(() -> BFLA._dot(Native, x, y, pval))
        println("dot   p=$p n=$n  public=", round(t_pub.median, sigdigits = 2),
                "s scan=", round(t_scan.median, sigdigits = 2),
                "s kernel=", round(t_kernel.median, sigdigits = 2),
                "s alloc_pub=", t_pub.alloc)
    end
end

for p in (128, 256)
    for n in (32, 128, 256)
        rng = MersenneTwister(8000 + p + n)
        A = owned_matrix(n, n, p, rng)
        B = owned_matrix(n, n, p, rng)
        C = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        one = BigFloat(1; precision = p)
        zero = BigFloat(0; precision = p)
        pval = BFLA._check_precision(one, zero, A, B, C)
        t_pub = measure(() -> BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, C))
        t_scan = measure(() -> BFLA._check_precision(one, zero, A, B, C))
        t_kernel = measure(() -> BFLA._gemm!(Native, Val(NoTrans), Val(NoTrans), one, A, B, zero, C, pval))
        println("gemm  p=$p n=$n  public=", round(t_pub.median, sigdigits = 2),
                "s scan=", round(t_scan.median, sigdigits = 2),
                "s kernel=", round(t_kernel.median, sigdigits = 2),
                "s alloc_pub=", t_pub.alloc)
    end
end

println("=== done ===")
