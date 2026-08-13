# Correctness-gated operation-only timing against the frozen SDPX BigFloat
# dense kernels. Mutable operands are rebuilt outside every timed sample.

include("bench_utils.jl")
using SparseArrays

function legacy_path()
    configured = get(ENV, "SDPX_LEGACY_BIGFLOAT", "")
    isempty(configured) || return configured
    candidate = joinpath(
        @__DIR__, "..", "..", "SDPX", "SDPX-v041-unified-la-probes",
        "src", "kernels", "bigfloat.jl",
    )
    isfile(candidate) || error(
        "SDPX legacy kernel file not found at $candidate; " *
        "set SDPX_LEGACY_BIGFLOAT",
    )
    return candidate
end

const LEGACY_PATH = normpath(legacy_path())
include(LEGACY_PATH)

const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (8, 32, 64, 128))
const PRECISIONS =
    parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "10"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))

SAMPLES >= 10 || error("BFLA_BENCH_SAMPLES must be at least 10")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")

function legacy_commit()
    repository = normpath(joinpath(dirname(LEGACY_PATH), "..", ".."))
    return try
        readchomp(`git -C $repository rev-parse HEAD`)
    catch
        "unavailable"
    end
end

function lower_equal(A, B)
    size(A) == size(B) || return false
    n = size(A, 1)
    size(A, 2) == n || return false
    @inbounds for j in 1:n, i in j:n
        A[i, j] == B[i, j] || return false
    end
    return true
end

function report_pair(workload, p, n, native_result, legacy_result)
    report_measurement(
        workload = workload,
        backend = "bfla_native",
        precision = p,
        size = n,
        threads = 1,
        block_size = 0,
        result = native_result,
    )
    report_measurement(
        workload = workload,
        backend = "sdpx_legacy",
        precision = p,
        size = n,
        threads = 1,
        block_size = nothing,
        result = legacy_result,
    )
    ratio = native_result.median / legacy_result.median
    println(
        "ratio workload=", workload,
        " precision=", p,
        " size=", n,
        " native_over_legacy=", round(ratio; digits = 4),
        " allocation_ratio=", round(
            native_result.allocated_bytes_median /
            max(legacy_result.allocated_bytes_median, 1);
            digits = 4,
        ),
    )
    return ratio
end

function benchmark_fixture(p, n, ratios)
    rng = MersenneTwister(9000 + p + n)
    one_value = BigFloat(1; precision = p)
    zero_value = BigFloat(0; precision = p)

    A = owned_matrix(n, n, p, rng)
    B = owned_matrix(n, n, p, rng)
    native_C = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    legacy_C = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.gemm!(
        Native, NoTrans, NoTrans, one_value, A, B, zero_value, native_C,
    )
    setprecision(BigFloat, p) do
        kmul_owned!(legacy_C, A, B, one_value, zero_value)
    end
    native_C == legacy_C || error("GEMM parity failed at p=$p n=$n")
    native_gemm = measure(
        () -> BFLA.gemm!(
            Native, NoTrans, NoTrans, one_value, A, B, zero_value, native_C,
        );
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> BFLA.zero_owned!(native_C),
    )
    legacy_gemm = measure(
        () -> setprecision(BigFloat, p) do
            kmul_owned!(legacy_C, A, B, one_value, zero_value)
        end;
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> BFLA.zero_owned!(legacy_C),
    )
    push!(ratios, report_pair("gemm", p, n, native_gemm, legacy_gemm))

    R = owned_matrix(n, n, p, rng)
    S = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.gemm!(Native, Trans, NoTrans, one_value, R, R, zero_value, S)
    @inbounds for i in 1:n
        BFLA.MA.operate!(+, S[i, i], one_value)
    end
    native_L = BFLA.owned_copy(S)
    native_factor = BFLA.cholesky!(Native, native_L)
    legacy_L = BFLA.owned_copy(S)
    legacy_success = setprecision(BigFloat, p) do
        kchol!(legacy_L)
    end
    (legacy_success && BFLA.issuccess(native_factor)) ||
        error("Cholesky success parity failed at p=$p n=$n")
    lower_equal(native_L, legacy_L) ||
        error("Cholesky factor parity failed at p=$p n=$n")
    native_state = Ref{Matrix{BigFloat}}()
    legacy_state = Ref{Matrix{BigFloat}}()
    native_cholesky = measure(
        () -> BFLA.cholesky!(Native, native_state[]);
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> (native_state[] = BFLA.owned_copy(S)),
    )
    legacy_cholesky = measure(
        () -> setprecision(BigFloat, p) do
            kchol!(legacy_state[])
        end;
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> (legacy_state[] = BFLA.owned_copy(S)),
    )
    push!(
        ratios,
        report_pair("cholesky", p, n, native_cholesky, legacy_cholesky),
    )

    cholesky_workspace = BFLA.BFLAWorkspace(p; workers = 1)
    workspace_L = BFLA.owned_copy(S)
    workspace_factor = BFLA.cholesky!(
        Native, workspace_L; workspace = cholesky_workspace,
    )
    BFLA.issuccess(workspace_factor) ||
        error("workspace Cholesky failed at p=$p n=$n")
    lower_equal(workspace_L, legacy_L) ||
        error("workspace Cholesky parity failed at p=$p n=$n")
    native_cholesky_workspace = measure(
        () -> BFLA.cholesky!(
            Native, native_state[]; workspace = cholesky_workspace,
        );
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> (native_state[] = BFLA.owned_copy(S)),
    )
    push!(
        ratios,
        report_pair(
            "cholesky_workspace",
            p,
            n,
            native_cholesky_workspace,
            legacy_cholesky,
        ),
    )

    triangular = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    @inbounds for j in 1:n, i in j:n
        BFLA.MA.operate_to!(
            triangular[i, j], copy,
            i == j ? BigFloat(n + i; precision = p) : A[i, j],
        )
    end
    X = owned_matrix(n, n, p, rng)
    native_X = BFLA.owned_copy(X)
    BFLA.trsm!(
        Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one_value,
        triangular, native_X,
    )
    legacy_X = BFLA.owned_copy(X)
    setprecision(BigFloat, p) do
        ktrsm!(triangular, legacy_X)
    end
    native_X == legacy_X || error("TRSM parity failed at p=$p n=$n")
    native_trsm = measure(
        () -> BFLA.trsm!(
            Native, LeftSide, Lower, NoTrans, NonUnitDiagonal, one_value,
            triangular, native_state[],
        );
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> (native_state[] = BFLA.owned_copy(X)),
    )
    legacy_trsm = measure(
        () -> setprecision(BigFloat, p) do
            ktrsm!(triangular, legacy_state[])
        end;
        warmup = WARMUP,
        samples = SAMPLES,
        setup = () -> (legacy_state[] = BFLA.owned_copy(X)),
    )
    push!(ratios, report_pair("trsm", p, n, native_trsm, legacy_trsm))

    x = vec(owned_matrix(n, 1, p, rng))
    y = vec(owned_matrix(n, 1, p, rng))
    native_dot_value = BFLA.dot(Native, x, y)
    legacy_dot_value = setprecision(BigFloat, p) do
        kdot(x, y)
    end
    native_dot_value == legacy_dot_value ||
        error("dot parity failed at p=$p n=$n")
    native_dot = measure(
        () -> BFLA.dot(Native, x, y);
        warmup = WARMUP,
        samples = SAMPLES,
    )
    legacy_dot = measure(
        () -> setprecision(BigFloat, p) do
            kdot(x, y)
        end;
        warmup = WARMUP,
        samples = SAMPLES,
    )
    push!(ratios, report_pair("dot", p, n, native_dot, legacy_dot))

    println("gate precision=$p size=$n status=passed")
end

println("=== BFLA Native vs frozen SDPX legacy ===")
println(environment())
println((legacy_path = LEGACY_PATH, legacy_commit = legacy_commit()))
println((precisions=PRECISIONS, sizes=SIZES, warmup=WARMUP, samples=SAMPLES))
ratios = Float64[]
for p in PRECISIONS, n in SIZES
    benchmark_fixture(p, n, ratios)
end
println((
    median_native_over_legacy = Statistics.median(ratios),
    max_native_over_legacy = maximum(ratios),
    min_native_over_legacy = minimum(ratios),
    cells_over_five_percent = count(>(1.05), ratios),
    cell_count = length(ratios),
    final_max_rss_bytes = Int(Sys.maxrss()),
))
println("=== done ===")
