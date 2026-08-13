# Correctness-gated standalone dense kernels and factorizations. Mutable
# operands are reset outside every timed sample through `measure(...; setup)`.

include("bench_utils.jl")

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (8, 16, 32, 64, 128, 256))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "10"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))
const THREAD_COUNT = parse(Int, get(ENV, "BFLA_BENCH_NATIVE_THREADS", "1"))
const BLOCK_SIZE = parse(Int, get(ENV, "BFLA_BENCH_BLOCK_SIZE", "0"))

SAMPLES >= 10 || error("BFLA_BENCH_SAMPLES must be at least 10")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")
THREAD_COUNT >= 1 || error("BFLA_BENCH_NATIVE_THREADS must be positive")
THREAD_COUNT <= Threads.nthreads() || error(
    "BFLA_BENCH_NATIVE_THREADS exceeds available Julia threads",
)
BLOCK_SIZE >= 0 || error("BFLA_BENCH_BLOCK_SIZE must be nonnegative")

function standalone_config(backend)
    backend === Native || return BFLA.KernelConfig()
    return BFLA.KernelConfig(
        thread_count = THREAD_COUNT,
        gemm_block = BLOCK_SIZE,
        syrk_block = BLOCK_SIZE,
        cholesky_block = BLOCK_SIZE,
        trsm_block = BLOCK_SIZE,
    )
end

function standalone_effective_threads(backend, workload)
    backend === Generic && return 1
    workload in ("cholesky", "cholesky_workspace", "ldlt", "rrqr") &&
        return 1
    return BLOCK_SIZE > 0 ? 1 : THREAD_COUNT
end

function standalone_residual_gate(A, X, B, p)
    residual = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
    BFLA.residual!(Native, A, X, B, residual)
    eta = BFLA.normwise_backward_error(Native, A, X, B, residual)
    @assert isfinite(eta)
    @assert eta <= benchmark_threshold(p, max(size(A)...))
    return eta
end

function standalone_fixture(p, n)
    rng = MersenneTwister(500_000 + 100p + n)
    A = owned_matrix(n, n, p, rng)
    B = owned_matrix(n, n, p, rng)
    R = owned_matrix(n, n, p, rng)
    spd = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(
        Native, Lower, NoTrans, BigFloat(1; precision = p), R,
        BigFloat(0; precision = p), spd,
    )
    BFLA.mirror_triangle!(spd, Lower)
    for i in 1:n
        BFLA.MA.operate!(+, spd[i, i], BigFloat(1; precision = p))
    end
    L = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in j:n
        BFLA.MA.operate_to!(
            L[i, j], copy,
            i == j ? BigFloat(n + i; precision = p) : A[i, j],
        )
    end
    indefinite = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for i in 1:n
        indefinite[i, i] = BigFloat(
            isodd(i) ? i + 2 : -(i + 2); precision = p,
        )
    end
    rectangular = owned_matrix(n, max(2, n ÷ 2), p, rng)
    Xtrue = owned_matrix(n, 2, p, rng)
    Bspd = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = p)
    Bindef = BFLA.owned_zeros(BigFloat, n, 2; precision_bits = p)
    BFLA.gemm!(
        Native, NoTrans, NoTrans, BigFloat(1; precision = p), spd,
        Xtrue, BigFloat(0; precision = p), Bspd,
    )
    BFLA.gemm!(
        Native, NoTrans, NoTrans, BigFloat(1; precision = p), indefinite,
        Xtrue, BigFloat(0; precision = p), Bindef,
    )

    one_value = BigFloat(1; precision = p)
    zero_value = BigFloat(0; precision = p)
    reference_gemm = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    reference_syrk = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    reference_trsm = BFLA.owned_copy(B)
    BFLA.gemm!(
        Generic, NoTrans, NoTrans, one_value, A, B, zero_value, reference_gemm,
    )
    BFLA.syrk!(
        Generic, Lower, NoTrans, one_value, A, zero_value, reference_syrk,
    )
    BFLA.trsm!(
        Generic, LeftSide, Lower, NoTrans, NonUnitDiagonal, one_value, L,
        reference_trsm,
    )

    for (backend_name, backend) in (("native", Native), ("generic", Generic))
        config = standalone_config(backend)
        Cgate = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        Sgate = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        Xgate = BFLA.owned_copy(B)
        BFLA.gemm!(
            backend, NoTrans, NoTrans, one_value, A, B, zero_value, Cgate;
            config = config,
        )
        BFLA.syrk!(
            backend, Lower, NoTrans, one_value, A, zero_value, Sgate;
            config = config,
        )
        BFLA.trsm!(
            backend, LeftSide, Lower, NoTrans, NonUnitDiagonal, one_value, L,
            Xgate; config = config,
        )
        @assert scaled_close(Cgate, reference_gemm, p; dimension = n)
        @assert scaled_close(Sgate, reference_syrk, p; dimension = n)
        @assert scaled_close(Xgate, reference_trsm, p; dimension = n)

        chol = BFLA.cholesky(backend, spd; config = config)
        chol_workspace_gate = BFLA.cholesky(
            backend,
            spd;
            config = config,
            workspace = BFLA.BFLAWorkspace(p; workers = 1),
        )
        @assert triangle_scaled_close(
            BFLA.factor_matrix(chol_workspace_gate),
            BFLA.factor_matrix(chol),
            p,
            Lower;
            dimension = n,
        )
        Xchol = BFLA.solve(chol, Bspd)
        eta_chol = standalone_residual_gate(spd, Xchol, Bspd, p)
        ldlt_factor = BFLA.ldlt(backend, indefinite)
        Xldlt = BFLA.solve(ldlt_factor, Bindef)
        eta_ldlt = standalone_residual_gate(indefinite, Xldlt, Bindef, p)
        rrqr = BFLA.qr(backend, rectangular)
        @assert BFLA.factor_rank(rrqr) == size(rectangular, 2)
        @assert BFLA.factor_diagnostics(rrqr).failure_position === nothing
        println(
            "gate backend=", backend_name,
            " precision=", p,
            " size=", n,
            " cholesky_eta=", eta_chol,
            " ldlt_eta=", eta_ldlt,
            " rrqr_rank=", BFLA.factor_rank(rrqr),
        )

        state = Ref{Matrix{BigFloat}}()
        cholesky_workspace = BFLA.BFLAWorkspace(p; workers = 1)
        measurements = (
            (
                "gemm",
                () -> BFLA.gemm!(
                    backend, NoTrans, NoTrans, one_value, A, B, zero_value,
                    state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_zeros(
                    BigFloat, n, n; precision_bits = p,
                )),
            ),
            (
                "syrk",
                () -> BFLA.syrk!(
                    backend, Lower, NoTrans, one_value, A, zero_value,
                    state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_zeros(
                    BigFloat, n, n; precision_bits = p,
                )),
            ),
            (
                "trsm",
                () -> BFLA.trsm!(
                    backend, LeftSide, Lower, NoTrans, NonUnitDiagonal,
                    one_value, L, state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_copy(B)),
            ),
            (
                "cholesky",
                () -> BFLA.cholesky!(backend, state[]; config = config),
                () -> (state[] = BFLA.owned_copy(spd)),
            ),
            (
                "cholesky_workspace",
                () -> BFLA.cholesky!(
                    backend,
                    state[];
                    config = config,
                    workspace = cholesky_workspace,
                ),
                () -> (state[] = BFLA.owned_copy(spd)),
            ),
            (
                "ldlt",
                () -> BFLA.ldlt!(backend, state[]),
                () -> (state[] = BFLA.owned_copy(indefinite)),
            ),
            (
                "rrqr",
                () -> BFLA.qr!(backend, state[]),
                () -> (state[] = BFLA.owned_copy(rectangular)),
            ),
        )
        for (workload, operation, setup) in measurements
            result = measure(
                operation; warmup = WARMUP, samples = SAMPLES, setup = setup,
            )
            report_measurement(
                workload = workload,
                backend = backend_name,
                precision = p,
                size = n,
                threads = standalone_effective_threads(backend, workload),
                block_size = backend === Native ? BLOCK_SIZE : nothing,
                result = result,
            )
        end
    end
end

println("=== BFLA standalone benchmark ===")
println(environment())
println((
    precisions = PRECISIONS,
    sizes = SIZES,
    warmup = WARMUP,
    samples = SAMPLES,
    native_threads = THREAD_COUNT,
    block_size = BLOCK_SIZE,
))
for p in PRECISIONS, n in SIZES
    standalone_fixture(p, n)
end
println((final_max_rss_bytes = Int(Sys.maxrss()),))
println("=== done ===")
