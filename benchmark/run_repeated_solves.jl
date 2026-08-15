# Correctness-gated repeated-solve benchmark for factor use boundaries.
#
# Workload matrix:
#   factorizations: Cholesky, LDLT, RRQR, LU
#   RHS shapes:     single vector, multi-RHS matrix
#   solve modes:    safe ldiv!, explicit ldiv_trusted!,
#                   ldiv_trusted! with caller-owned BFLAWorkspace scratch
#
# This runner targets the post-audit solve API:
#   ldiv!(F, rhs; workspace=nothing, workspace_worker=1)
#   ldiv_trusted!(F, rhs; workspace=nothing, workspace_worker=1)
#
# Configure with:
#   BFLA_BENCH_PRECISIONS=128,256,512
#   BFLA_BENCH_SIZES=8,16,32,64
#   BFLA_BENCH_NRHS=3
#   BFLA_BENCH_SAMPLES=10
#   BFLA_BENCH_WARMUP=2
#   BFLA_SOURCE_COMMIT=$(git rev-parse HEAD)

include("bench_utils.jl")
import MutableArithmetics as MA

const PRECISIONS = parse_int_tuple("BFLA_BENCH_PRECISIONS", (128, 256, 512))
const SIZES = parse_int_tuple("BFLA_BENCH_SIZES", (8, 16, 32, 64))
const NRHS = parse(Int, get(ENV, "BFLA_BENCH_NRHS", "3"))
const SAMPLES = parse(Int, get(ENV, "BFLA_BENCH_SAMPLES", "10"))
const WARMUP = parse(Int, get(ENV, "BFLA_BENCH_WARMUP", "2"))

SAMPLES >= 10 || error("BFLA_BENCH_SAMPLES must be at least 10")
WARMUP >= 2 || error("BFLA_BENCH_WARMUP must be at least 2")
NRHS >= 1 || error("BFLA_BENCH_NRHS must be positive")
all(>(0), PRECISIONS) || error("benchmark precisions must be positive")
all(>(0), SIZES) || error("benchmark sizes must be positive")

# Fixtures are exact rationals and integers staged directly into BigFloat at
# the target precision; no Float64 value is used in fixture construction.
function identity_shift!(A, p)
    one_value = BigFloat(1; precision = p)
    @inbounds for i in axes(A, 1)
        MA.operate!(+, A[i, i], one_value)
    end
    return A
end

function spd_fixture(n, p, rng)
    source = owned_matrix(n, n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(
        Native,
        Lower,
        NoTrans,
        BigFloat(1; precision = p),
        source,
        BigFloat(0; precision = p),
        A,
    )
    BFLA.mirror_triangle!(A, Lower)
    identity_shift!(A, p)
    return A
end

function indefinite_fixture(n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    @inbounds for j in 1:n, i in j:n
        value = if i == j
            BigFloat(isodd(i) ? i + 2 : -(i + 2); precision = p)
        else
            BigFloat(rand(rng, -8:8) // 16; precision = p)
        end
        A[i, j] = value
        A[j, i] = i == j ? value : BigFloat(value; precision = p)
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

function rhs_from_solution(A, X, p)
    one_value = BigFloat(1; precision = p)
    zero_value = BigFloat(0; precision = p)
    if X isa AbstractVector
        B = BFLA.owned_zeros(BigFloat, length(X); precision_bits = p)
        BFLA.gemv!(Native, NoTrans, one_value, A, X, zero_value, B)
    else
        B = BFLA.owned_zeros(BigFloat, size(X)...; precision_bits = p)
        BFLA.gemm!(
            Native, NoTrans, NoTrans, one_value, A, X, zero_value, B,
        )
    end
    return B
end

function residual_gate(A, X, B, p)
    residual = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
    BFLA.residual!(Native, A, X, B, residual)
    eta = BFLA.normwise_backward_error(Native, A, X, B, residual)
    @assert isfinite(eta)
    @assert eta <= benchmark_threshold(p, max(size(A)...))
    return eta
end

function safe_solution(F, B, p)
    X = BFLA.owned_copy(B)
    BFLA.ldiv!(F, X)
    @assert all(isfinite, X)
    return X
end

function trusted_solution(F, B, p; workspace = nothing)
    X = BFLA.owned_copy(B)
    if workspace === nothing
        BFLA.ldiv_trusted!(F, X)
    else
        BFLA.ldiv_trusted!(
            F, X; workspace = workspace, workspace_worker = 1,
        )
    end
    @assert all(isfinite, X)
    return X
end

function gate_factorization(A, Fn, Fg, B, p, n, workspace)
    Xn = safe_solution(Fn, B, p)
    Xg = safe_solution(Fg, B, p)
    @assert scaled_close(Xn, Xg, p; dimension = n)
    eta_n = residual_gate(A, Xn, B, p)
    eta_g = residual_gate(A, Xg, B, p)

    for (backend_name, F, reference) in (
        ("native", Fn, Xn),
        ("generic", Fg, Xg),
    )
        trusted = trusted_solution(F, B, p)
        @assert scaled_close(trusted, reference, p; dimension = n)
        residual_gate(A, trusted, B, p)

        trusted_ws = trusted_solution(F, B, p; workspace = workspace)
        @assert scaled_close(trusted_ws, reference, p; dimension = n)
        residual_gate(A, trusted_ws, B, p)
    end
    return (native = eta_n, generic = eta_g)
end

function report_gate(name, shape, backend_name, p, n; eta)
    println(
        "gate workload=", name, "_", shape,
        " backend=", backend_name,
        " precision=", p,
        " size=", n,
        " backward_error=", eta,
        " status=passed",
    )
end

function measure_solve_modes(backend_name, workload, factor, B, n, p, workspace)
    state = Ref{AbstractVecOrMat{BigFloat}}()
    modes = (
        ("safe", () -> BFLA.ldiv!(factor, state[])),
        ("trusted", () -> BFLA.ldiv_trusted!(factor, state[])),
        ("trusted_ws", () -> BFLA.ldiv_trusted!(
            factor, state[]; workspace = workspace, workspace_worker = 1,
        )),
    )
    for (mode, operation) in modes
        result = measure(
            operation;
            warmup = WARMUP,
            samples = SAMPLES,
            setup = () -> (state[] = BFLA.owned_copy(B)),
        )
        report_measurement(
            workload = "$(workload)_$(mode)",
            backend = backend_name,
            precision = p,
            size = n,
            threads = 1,
            block_size = nothing,
            result = result,
        )
    end
    return nothing
end

function benchmark_fixture(p, n)
    rng = MersenneTwister(600_000 + 100p + n)
    spd = spd_fixture(n, p, rng)
    indefinite = indefinite_fixture(n, p, rng)
    square = square_fixture(n, p, rng)
    Xtrue_vec = vec(owned_matrix(n, 1, p, rng))
    Xtrue_mat = owned_matrix(n, NRHS, p, rng)

    factor_sets = (
        (
            "chol", spd,
            BFLA.cholesky(Native, spd), BFLA.cholesky(Generic, spd),
            rhs_from_solution(spd, Xtrue_vec, p),
            rhs_from_solution(spd, Xtrue_mat, p),
        ),
        (
            "ldlt", indefinite,
            BFLA.ldlt(Native, indefinite), BFLA.ldlt(Generic, indefinite),
            rhs_from_solution(indefinite, Xtrue_vec, p),
            rhs_from_solution(indefinite, Xtrue_mat, p),
        ),
        (
            "lu", square,
            BFLA.lu(Native, square), BFLA.lu(Generic, square),
            rhs_from_solution(square, Xtrue_vec, p),
            rhs_from_solution(square, Xtrue_mat, p),
        ),
        (
            "rrqr", square,
            BFLA.qr(Native, square), BFLA.qr(Generic, square),
            rhs_from_solution(square, Xtrue_vec, p),
            rhs_from_solution(square, Xtrue_mat, p),
        ),
    )

    for (_, _, Fn, Fg, _, _) in factor_sets
        @assert BFLA.issuccess(Fn) && BFLA.issuccess(Fg)
    end
    _, _, native_ldlt, generic_ldlt, _, _ = factor_sets[2]
    @assert BFLA.factor_inertia(native_ldlt) ==
            BFLA.factor_inertia(generic_ldlt)
    _, _, native_qr, generic_qr, _, _ = factor_sets[4]
    @assert BFLA.factor_rank(native_qr) == n
    @assert BFLA.factor_rank(generic_qr) == n

    workspace = BFLA.BFLAWorkspace(p; workers = 1)
    for (name, A, Fn, Fg, Bvec, Bmat) in factor_sets
        for (shape, B) in (("vector", Bvec), ("multirhs", Bmat))
            etas = gate_factorization(A, Fn, Fg, B, p, n, workspace)
            report_gate(name, shape, "native", p, n; eta = etas.native)
            report_gate(name, shape, "generic", p, n; eta = etas.generic)
        end
    end

    for (backend_name, backend) in (("native", Native), ("generic", Generic))
        for (name, _, Fn, Fg, Bvec, Bmat) in factor_sets
            factor = backend === Native ? Fn : Fg
            measure_solve_modes(
                backend_name, "$(name)_vector", factor, Bvec, n, p,
                workspace,
            )
            measure_solve_modes(
                backend_name, "$(name)_multirhs", factor, Bmat, n, p,
                workspace,
            )
        end
    end
    return nothing
end

println("=== BFLA repeated-solve benchmark ===")
println(environment())
println((
    precisions = PRECISIONS,
    sizes = SIZES,
    nrhs = NRHS,
    warmup = WARMUP,
    samples = SAMPLES,
))
for p in PRECISIONS, n in SIZES
    benchmark_fixture(p, n)
end
println((final_max_rss_bytes = Int(Sys.maxrss()),))
println("=== done ===")
