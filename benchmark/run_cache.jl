# Phase-5 reusable-cache benchmark: runtime, zero-allocation, and RSS stability
# for the warm cache hot path (factorize! + solve_trusted!). Residual and
# refinement are NOT measured here; their allocation is reported separately in
# docs/src/memory_accounting.md and test/caches.jl.
#
# Reports Julia-allocated bytes per single call after warm-up, median runtime,
# and the Sys.maxrss delta across repeated cycles. A stable RSS delta (0) is the
# honest signal that no new native (MPFR) blocks are retained.
#
# Configure:
#   BFLA_BENCH_PRECISIONS=128,256,512
#   BFLA_BENCH_SIZES=8,32,128
#   BFLA_BENCH_NRHS=3
#   BFLA_BENCH_SAMPLES=5
#   BFLA_BENCH_WARMUP=8

include("bench_utils.jl")
import MutableArithmetics as MA

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (8, 32, 128))
const NRHS = parse(Int, get(ENV, "BFLA_BENCH_NRHS", "3"))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "5"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "8"))

function spd_fixture(n, p, rng)
    S = owned_matrix(n, n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(
        Native, Lower, NoTrans, BigFloat(1; precision = p), S,
        BigFloat(0; precision = p), A,
    )
    BFLA.mirror_triangle!(A, Lower)
    one_value = BigFloat(1; precision = p)
    @inbounds for i in axes(A, 1)
        MA.operate!(+, A[i, i], one_value)
    end
    return A
end

function square_fixture(n, p, rng)
    A = owned_matrix(n, n, p, rng)
    @inbounds for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(n + i; precision = p))
    end
    return A
end

function measure_cache(name, p, cache, A, B)
    x = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
    BFLA.factorize!(cache, A)
    BFLA.solve_trusted!(x, cache, B)
    for _ in 1:WARMUP
        BFLA.factorize!(cache, A)
        BFLA.solve_trusted!(x, cache, B)
    end
    rss_before = Int(Sys.maxrss())
    solve_alloc = @allocated BFLA.solve_trusted!(x, cache, B)
    factor_alloc = @allocated BFLA.factorize!(cache, A)
    cycle_alloc = @allocated begin
        BFLA.factorize!(cache, A)
        BFLA.solve_trusted!(x, cache, B)
    end
    times = Vector{Float64}(undef, SAMPLES)
    for s in 1:SAMPLES
        GC.gc()
        times[s] = @elapsed begin
            BFLA.factorize!(cache, A)
            BFLA.solve_trusted!(x, cache, B)
        end
    end
    rss_after = Int(Sys.maxrss())
    println(
        "cache workload=", name, " precision=", p, " size=", size(A, 1),
        " solve_alloc_bytes=", solve_alloc,
        " factorize_alloc_bytes=", factor_alloc,
        " cycle_alloc_bytes=", cycle_alloc,
        " cycle_median_s=", round(Statistics.median(times); sigdigits = 6),
        " rss_delta_bytes=", max(rss_after - rss_before, 0),
    )
    return nothing
end

println("=== BFLA reusable-cache benchmark ===")
println(environment())
for p in PRECISIONS, n in SIZES
    rng = MersenneTwister(700_000 + 100p + n)
    spd = spd_fixture(n, p, rng)
    square = square_fixture(n, p, rng)
    Bspd = BFLA.owned_zeros(BigFloat, n, NRHS; precision_bits = p)
    for j in 1:NRHS, i in 1:n
        Bspd[i, j] = BigFloat(rand(rng, -1024:1024); precision = p)
    end
    ch = BFLA.BFLACholeskyCache(Native)
    BFLA.prepare!(ch, n, p; nrhs = NRHS)
    measure_cache("cholesky", p, ch, spd, Bspd)
    lu = BFLA.BFLALUCache(Native)
    BFLA.prepare!(lu, n, p; nrhs = NRHS)
    measure_cache("lu", p, lu, square, Bspd)
end
println((
    final_max_rss_bytes = Int(Sys.maxrss()),
))
println("=== done ===")
