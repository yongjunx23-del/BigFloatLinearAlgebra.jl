# Correctness-gated solver-like dense cycle benchmark:
#
#   SYRK assembly -> Cholesky -> two-RHS solve -> q-bit residual
#                 -> one explicit refinement step
#
# Native and Generic consume the same immutable baseline fixtures. Every timed
# sample reconstructs all operands that its stage mutates.

include("bench_utils.jl")
import MutableArithmetics as MA

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (16, 32, 64))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "10"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))
const THREAD_COUNT = parse(Int, get(ENV, "BFLA_BENCH_NATIVE_THREADS", "1"))

SAMPLES >= 10 || error("BFLA_BENCH_SAMPLES must be at least 10")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")

function assemble(backend, source, p)
    n = size(source, 1)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    one_value = BigFloat(1; precision = p)
    zero_value = BigFloat(0; precision = p)
    config = backend === Native ? BFLA.KernelConfig(
        thread_count = THREAD_COUNT,
    ) : BFLA.KernelConfig()
    BFLA.syrk!(
        backend,
        Lower,
        NoTrans,
        one_value,
        source,
        zero_value,
        A;
        config = config,
    )
    BFLA.mirror_triangle!(A, Lower)
    @inbounds for i in 1:n
        MA.operate!(+, A[i, i], one_value)
    end
    return A
end

function solve_fixture(backend, A, B)
    factor = BFLA.cholesky(backend, A)
    X = BFLA.owned_copy(B)
    BFLA.solve!(factor, X)
    return factor, X
end

function dense_cycle(backend, source, B, p, q)
    A = assemble(backend, source, p)
    factor, X = solve_fixture(backend, A, B)
    residual = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = q)
    before = BFLA.higher_precision_residual!(
        backend,
        A,
        X,
        B,
        residual;
        residual_precision = q,
        factor_precision = p,
    )
    correction = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
    refinement = BFLA.refine_once!(
        factor, A, X, B, residual, correction,
    )
    return (
        X = X,
        backward_error_before = before.backward_error,
        backward_error_after = refinement.backward_error_after,
    )
end

function report_stage(backend_name, stage, p, n, result)
    println(
        "backend=", backend_name,
        " stage=", stage,
        " p=", p,
        " n=", n,
        " median_s=", round(result.median; sigdigits = 6),
        " iqr_s=", round(result.iqr; sigdigits = 6),
        " min_s=", round(result.min; sigdigits = 6),
        " max_s=", round(result.max; sigdigits = 6),
        " median_bytes=", result.allocs,
    )
end

function benchmark_fixture(p, n)
    q = 2p
    rng = MersenneTwister(100_000 + p + n)
    source = owned_matrix(n, n, p, rng)
    Xtrue = owned_matrix(n, 2, p, rng)
    A_baseline = assemble(Native, source, p)
    B = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = p)
    BFLA.gemm!(
        Native,
        NoTrans,
        NoTrans,
        BigFloat(1; precision = p),
        A_baseline,
        Xtrue,
        BigFloat(0; precision = p),
        B,
    )

    for (name, backend) in (("native", Native), ("generic", Generic))
        gate = dense_cycle(backend, source, B, p, q)
        threshold = BigFloat(10_000n; precision = p) * eps_bits(p)
        solution_error = maximum(
            abs(gate.X[index] - Xtrue[index]) for index in eachindex(Xtrue)
        )
        @assert isfinite(gate.backward_error_before)
        @assert isfinite(gate.backward_error_after)
        @assert gate.backward_error_before <= threshold
        @assert gate.backward_error_after <= threshold
        @assert solution_error <= threshold
        println(
            "gate backend=", name,
            " p=", p,
            " n=", n,
            " eta_before=", gate.backward_error_before,
            " eta_after=", gate.backward_error_after,
            " solution_error_inf=", solution_error,
        )

        A = assemble(backend, source, p)
        factor, X = solve_fixture(backend, A, B)
        assembly = measure(
            () -> assemble(backend, source, p);
            warmup = WARMUP,
            samples = SAMPLES,
        )
        factorization = measure(
            () -> BFLA.cholesky(backend, A);
            warmup = WARMUP,
            samples = SAMPLES,
        )
        solve_time = measure(
            () -> begin
                Y = BFLA.owned_copy(B)
                BFLA.solve!(factor, Y)
            end;
            warmup = WARMUP,
            samples = SAMPLES,
        )
        residual_time = measure(
            () -> begin
                R = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = q)
                BFLA.higher_precision_residual!(
                    backend,
                    A,
                    X,
                    B,
                    R;
                    residual_precision = q,
                    factor_precision = p,
                )
            end;
            warmup = WARMUP,
            samples = SAMPLES,
        )
        refinement_time = measure(
            () -> begin
                Y = BFLA.owned_copy(X)
                R = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = q)
                D = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = p)
                BFLA.refine_once!(factor, A, Y, B, R, D)
            end;
            warmup = WARMUP,
            samples = SAMPLES,
        )
        total = measure(
            () -> dense_cycle(backend, source, B, p, q);
            warmup = WARMUP,
            samples = SAMPLES,
        )

        report_stage(name, "assembly", p, n, assembly)
        report_stage(name, "factor", p, n, factorization)
        report_stage(name, "solve_2rhs", p, n, solve_time)
        report_stage(name, "residual_q", p, n, residual_time)
        report_stage(name, "refine_once", p, n, refinement_time)
        report_stage(name, "total_cycle", p, n, total)
    end
end

println("=== BFLA solver-like dense cycle benchmark ===")
println(environment())
println((precisions = PRECISIONS, sizes = SIZES, warmup = WARMUP,
         samples = SAMPLES, native_threads = THREAD_COUNT))
for p in PRECISIONS, n in SIZES
    benchmark_fixture(p, n)
end
println((final_max_rss_bytes = Sys.maxrss(),))
println("=== done ===")
