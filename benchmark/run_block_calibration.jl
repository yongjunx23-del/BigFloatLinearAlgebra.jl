# Explicit block/thread calibration. BFLA reports behavior; callers choose
# their own KernelConfig policy.

include("bench_utils.jl")

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (16, 32, 64, 128, 256))
const BLOCK_SIZES = parse_int_tuple(
    "BFLA_BENCH_BLOCK_SIZES", (0, 8, 16, 24, 32, 48, 64),
)
const THREAD_COUNTS = parse_int_tuple("BFLA_BENCH_THREAD_COUNTS", (1,))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "10"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))

SAMPLES >= 10 || error("BFLA_BENCH_SAMPLES must be at least 10")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")
all(>(0), PRECISIONS) || error("benchmark precisions must be positive")
all(>(0), SIZES) || error("benchmark sizes must be positive")
all(>=(0), BLOCK_SIZES) || error("benchmark block sizes must be nonnegative")
all(>(0), THREAD_COUNTS) || error("benchmark thread counts must be positive")
all(<=(Threads.nthreads()), THREAD_COUNTS) || error(
    "requested benchmark thread count exceeds available Julia threads",
)
any(>(0), BLOCK_SIZES) && !(1 in THREAD_COUNTS) && error(
    "blocked calibration requires thread count 1",
)

function calibration_fixture(p, n)
    rng = MersenneTwister(400_000 + 100p + n)
    A = owned_matrix(n, n, p, rng)
    B = owned_matrix(n, n, p, rng)
    R = owned_matrix(n, n, p, rng)
    spd = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(Native, Lower, NoTrans, BigFloat(1; precision = p), R,
               BigFloat(0; precision = p), spd)
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
    one_value = BigFloat(1; precision = p)
    zero_value = BigFloat(0; precision = p)
    gemm_reference = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    syrk_reference = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    trsm_reference = BFLA.owned_copy(B)
    BFLA.gemm!(Generic, NoTrans, NoTrans, one_value, A, B, zero_value,
               gemm_reference)
    BFLA.syrk!(Generic, Lower, NoTrans, one_value, A, zero_value,
               syrk_reference)
    BFLA.trsm!(Generic, LeftSide, Lower, NoTrans, NonUnitDiagonal,
               one_value, L, trsm_reference)

    for threads in THREAD_COUNTS, block in BLOCK_SIZES
        # Blocked dispatch currently takes precedence over threaded dispatch,
        # so threads > 1 with block > 0 is a duplicate single-threaded run.
        block > 0 && threads > 1 && continue
        config = BFLA.KernelConfig(
            thread_count = threads,
            gemm_block = block,
            syrk_block = block,
            cholesky_block = block,
            trsm_block = block,
        )
        gemm_gate = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        syrk_gate = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        trsm_gate = BFLA.owned_copy(B)
        BFLA.gemm!(Native, NoTrans, NoTrans, one_value, A, B, zero_value,
                   gemm_gate; config = config)
        BFLA.syrk!(Native, Lower, NoTrans, one_value, A, zero_value,
                   syrk_gate; config = config)
        BFLA.trsm!(Native, LeftSide, Lower, NoTrans, NonUnitDiagonal,
                   one_value, L, trsm_gate; config = config)
        cholesky_gate = BFLA.cholesky(Native, spd; config = config)
        @assert scaled_close(gemm_gate, gemm_reference, p; dimension = n)
        @assert scaled_close(syrk_gate, syrk_reference, p; dimension = n)
        @assert scaled_close(trsm_gate, trsm_reference, p; dimension = n)
        @assert BFLA.issuccess(cholesky_gate)
        println("gate precision=", p, " size=", n, " threads=", threads,
                " block_size=", block,
                " dispatch=", block > 0 ? "blocked_single" :
                    threads > 1 ? "threaded_unblocked" : "serial_unblocked",
                " status=passed")

        state = Ref{Matrix{BigFloat}}()
        for (name, operation, setup) in (
            (
                "gemm",
                () -> BFLA.gemm!(
                    Native, NoTrans, NoTrans, one_value, A, B, zero_value,
                    state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_zeros(
                    BigFloat, n, n; precision_bits = p,
                )),
            ),
            (
                "syrk",
                () -> BFLA.syrk!(
                    Native, Lower, NoTrans, one_value, A, zero_value,
                    state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_zeros(
                    BigFloat, n, n; precision_bits = p,
                )),
            ),
            (
                "trsm",
                () -> BFLA.trsm!(
                    Native, LeftSide, Lower, NoTrans, NonUnitDiagonal,
                    one_value, L, state[]; config = config,
                ),
                () -> (state[] = BFLA.owned_copy(B)),
            ),
            (
                "cholesky",
                () -> BFLA.cholesky!(Native, state[]; config = config),
                () -> (state[] = BFLA.owned_copy(spd)),
            ),
        )
            name == "cholesky" && threads > 1 && continue
            result = measure(
                operation; warmup = WARMUP, samples = SAMPLES, setup = setup,
            )
            report_measurement(
                workload = name,
                backend = "native",
                precision = p,
                size = n,
                threads = threads,
                block_size = block,
                result = result,
            )
        end
    end
end

println("=== BFLA block/thread calibration ===")
println(environment())
println((precisions=PRECISIONS, sizes=SIZES, blocks=BLOCK_SIZES,
         thread_counts=THREAD_COUNTS, warmup=WARMUP, samples=SAMPLES))
for p in PRECISIONS, n in SIZES
    calibration_fixture(p, n)
end
println((final_max_rss_bytes = Int(Sys.maxrss()),))
println("=== done ===")
