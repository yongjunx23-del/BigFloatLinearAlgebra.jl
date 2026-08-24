# Phase-1 numeric/memory baseline for the allocating (pre-cache) factor API.
#
# Records, per precision x size, the solver-grade facts the refactor targets:
#   runtime, Julia allocated bytes, RSS growth (Sys.maxrss), factor residual,
#   normwise backward error, factor precision, and vector/multi-RHS solves.
#
# Julia and native allocation are reported separately. We do NOT read
# `@allocated == 0` as evidence that MPFR/C-side allocation is zero; native
# allocation is bounded via Sys.maxrss deltas. A *stable* RSS across repeated
# cycles is the honest signal that no new native blocks are retained.
#
# Configure:
#   BFLA_BENCH_PRECISIONS=128,256,512
#   BFLA_BENCH_SIZES=8,32,128
#   BFLA_BENCH_NRHS=3
#   BFLA_BENCH_SAMPLES=8
#   BFLA_BENCH_WARMUP=2

include("bench_utils.jl")
import MutableArithmetics as MA

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (8, 32, 128))
const NRHS = parse(Int, get(ENV, "BFLA_BENCH_NRHS", "3"))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "8"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))

SAMPLES >= 5 || error("BFLA_BENCH_SAMPLES must be at least 5")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")

# --- fixtures (exact rationals/ints staged into BigFloat; no Float64) ---

function spd_fixture(n, p, rng)
    source = owned_matrix(n, n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(
        Native, Lower, NoTrans,
        BigFloat(1; precision = p), source,
        BigFloat(0; precision = p), A,
    )
    BFLA.mirror_triangle!(A, Lower)
    one_value = BigFloat(1; precision = p)
    @inbounds for i in axes(A, 1)
        MA.operate!(+, A[i, i], one_value)
    end
    return A
end

function square_fixture(n, p, r)
    A = owned_matrix(n, n, p, r)
    @inbounds for i in 1:n
        MA.operate!(+, A[i, i], BigFloat(n + i; precision = p))
    end
    return A
end

function rhs_from_solution(A, X, p)
    B = BFLA.owned_zeros(BigFloat, size(X)...; precision_bits = p)
    if X isa AbstractVector
        BFLA.gemv!(
            Native, NoTrans, BigFloat(1; precision = p), A, X,
            BigFloat(0; precision = p), B,
        )
    else
        BFLA.gemm!(
            Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, X,
            BigFloat(0; precision = p), B,
        )
    end
    return B
end

# Complete factor-solve-residual cycle on the allocating API; returns the
# Julia-allocated bytes, median runtime, and normwise backward error.
function measure_cycle(factor_ctor, A, B, p, n)
    rss_before = Int(Sys.maxrss())
    operation = function ()
        F = factor_ctor(A)
        X = BFLA.owned_copy(B)
        BFLA.ldiv!(F, X)
        residual = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
        BFLA.residual!(Native, A, X, B, residual)
        BFLA.normwise_backward_error(Native, A, X, B, residual)
    end
    result = measure(operation; warmup = WARMUP, samples = SAMPLES)
    rss_after = Int(Sys.maxrss())
    return (
        runtime_ns = result.median * 1e9,
        allocated_bytes = result.allocated_bytes_median,
        rss_delta_bytes = max(rss_after - rss_before, 0),
        backward_error = operation(),
        cold_rss_delta = result.cold_rss_delta,
    )
end

function baseline_one(p, n)
    r = MersenneTwister(900_000 + 100p + n)
    spd = spd_fixture(n, p, r)
    square = square_fixture(n, p, r)
    Xvec = vec(owned_matrix(n, 1, p, r))
    Xmat = owned_matrix(n, NRHS, p, r)
    Bspd_vec = rhs_from_solution(spd, Xvec, p)
    Bspd_mat = rhs_from_solution(spd, Xmat, p)
    Bsq_vec = rhs_from_solution(square, Xvec, p)
    Bsq_mat = rhs_from_solution(square, Xmat, p)

    for (kind, A, Bv, Bm) in (
        ("cholesky", spd, Bspd_vec, Bspd_mat),
        ("lu", square, Bsq_vec, Bsq_mat),
    )
        ctor = kind == "cholesky" ? BFLA.cholesky : BFLA.lu
        for (shape, B) in (("vector", Bv), ("multirhs", Bm))
            m = measure_cycle((A) -> ctor(Native, A), A, B, p, n)
            F = ctor(Native, A)
            println(
                "baseline backend=native kind=", kind, " shape=", shape,
                " precision=", p, " size=", n,
                " runtime_ns=", round(m.runtime_ns; digits = 1),
                " julia_allocated_bytes=", m.allocated_bytes,
                " rss_delta_bytes=", m.rss_delta_bytes,
                " cold_rss_delta_bytes=", m.cold_rss_delta,
                " factor_precision=", BFLA.factor_precision(F),
                " factor_kind=", BFLA.factor_kind(F),
                " backward_error=", m.backward_error,
            )
        end
    end
    return nothing
end

println("=== BFLA allocating-factor baseline ===")
println(environment())
println((
    precisions = PRECISIONS,
    sizes = SIZES,
    nrhs = NRHS,
    warmup = WARMUP,
    samples = SAMPLES,
))
for p in PRECISIONS, n in SIZES
    baseline_one(p, n)
end
println((
    final_max_rss_bytes = Int(Sys.maxrss()),
))
println("=== done ===")
