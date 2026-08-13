# Correctness-gated production combinations used by dense BigFloat callers.
# Every timed invocation reconstructs operands mutated by its workload.

include("bench_utils.jl")
import MutableArithmetics as MA

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

function config(backend)
    backend === Native || return BFLA.KernelConfig()
    return BFLA.KernelConfig(
        thread_count = THREAD_COUNT,
        gemm_block = BLOCK_SIZE,
        syrk_block = BLOCK_SIZE,
        cholesky_block = BLOCK_SIZE,
        trsm_block = BLOCK_SIZE,
    )
end

function cycle_effective_threads(backend, workload)
    backend === Generic && return 1
    workload in (
        "cholesky_three_solves",
        "cholesky_three_solves_workspace",
        "ldlt_multi_rhs",
    ) && return 1
    return BLOCK_SIZE > 0 ? 1 : THREAD_COUNT
end

function identity_shift!(A, p)
    one_value = BigFloat(1; precision = p)
    @inbounds for i in axes(A, 1)
        MA.operate!(+, A[i, i], one_value)
    end
    return A
end

function spd_from_source(backend, source, p)
    n = size(source, 1)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.syrk!(
        backend,
        Lower,
        NoTrans,
        BigFloat(1; precision = p),
        source,
        BigFloat(0; precision = p),
        A;
        config = config(backend),
    )
    BFLA.mirror_triangle!(A, Lower)
    identity_shift!(A, p)
    return A
end

function triangular_fixture(n, p, rng)
    L = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    @inbounds for j in 1:n, i in j:n
        L[i, j] = i == j ? BigFloat(n + i; precision = p) :
                  BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    return L
end

function indefinite_fixture(n, p)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    @inbounds for i in 1:n
        A[i, i] = BigFloat(isodd(i) ? i + 2 : -(i + 2); precision = p)
    end
    # Small symmetric coupling prevents the fixture from degenerating into a
    # diagonal-only benchmark while preserving a deterministic nonsingular D.
    coupling = BigFloat(1 // 16; precision = p)
    @inbounds for i in 2:n
        value = isodd(i) ? coupling : -coupling
        MA.operate_to!(A[i, i - 1], copy, value)
        MA.operate_to!(A[i - 1, i], copy, value)
    end
    return A
end

function rhs_from_solution(A, X, p)
    B = BFLA.owned_zeros(BigFloat, size(X)...; precision_bits = p)
    BFLA.gemm!(
        Native,
        NoTrans,
        NoTrans,
        BigFloat(1; precision = p),
        A,
        X,
        BigFloat(0; precision = p),
        B,
    )
    return B
end

function gate_solution(A, X, B, p)
    residual = BFLA.owned_zeros(BigFloat, size(B)...; precision_bits = p)
    BFLA.residual!(Native, A, X, B, residual)
    eta = BFLA.normwise_backward_error(Native, A, X, B, residual)
    threshold = benchmark_threshold(p, max(size(A)...))
    @assert isfinite(eta) && eta <= threshold
    return eta
end

function cycle_syrk_cholesky(backend, source, p; workspace=nothing)
    A = spd_from_source(backend, source, p)
    factor = BFLA.cholesky!(
        backend, A; config = config(backend), workspace = workspace,
    )
    return factor
end

function cycle_cholesky_three_solves(backend, A, B; workspace=nothing)
    factor = BFLA.cholesky(
        backend, A; config = config(backend), workspace = workspace,
    )
    results = ntuple(_ -> begin
        X = BFLA.owned_copy(B)
        BFLA.solve!(factor, X)
        X
    end, 3)
    return factor, results
end

function cycle_trsm_gram_rrqr(backend, L, B, p)
    X = BFLA.owned_copy(B)
    BFLA.trsm!(
        backend,
        LeftSide,
        Lower,
        NoTrans,
        NonUnitDiagonal,
        BigFloat(1; precision = p),
        L,
        X;
        config = config(backend),
    )
    gram = BFLA.owned_zeros(BigFloat, size(X, 2), size(X, 2); precision_bits = p)
    BFLA.syrk!(
        backend,
        Lower,
        Trans,
        BigFloat(1; precision = p),
        X,
        BigFloat(0; precision = p),
        gram;
        config = config(backend),
    )
    BFLA.mirror_triangle!(gram, Lower)
    factor = BFLA.qr(backend, gram)
    return X, gram, factor
end

function cycle_ldlt_multi_rhs(backend, A, B)
    factor = BFLA.ldlt(backend, A)
    X = BFLA.owned_copy(B)
    BFLA.solve!(factor, X)
    return factor, X
end

function report_gate(name, backend_name, p, n; kwargs...)
    print("gate workload=", name, " backend=", backend_name,
          " precision=", p, " size=", n)
    for (key, value) in pairs(kwargs)
        print(" ", key, "=", value)
    end
    println()
end

function benchmark_fixture(p, n)
    rng = MersenneTwister(300_000 + 100p + n)
    source = owned_matrix(n, n, p, rng)
    spd = spd_from_source(Native, source, p)
    Xtrue = owned_matrix(n, 3, p, rng)
    Bspd = rhs_from_solution(spd, Xtrue, p)
    L = triangular_fixture(n, p, rng)
    rectangular = owned_matrix(n, max(2, n ÷ 2), p, rng)
    indefinite = indefinite_fixture(n, p)
    Xindef = owned_matrix(n, 3, p, rng)
    Bindef = rhs_from_solution(indefinite, Xindef, p)

    native_chol = cycle_syrk_cholesky(Native, source, p)
    generic_chol = cycle_syrk_cholesky(Generic, source, p)
    @assert BFLA.issuccess(native_chol) && BFLA.issuccess(generic_chol)
    @assert triangle_scaled_close(
        BFLA.factor_matrix(native_chol), BFLA.factor_matrix(generic_chol), p,
        Lower; dimension = n,
    )

    _, native_solved = cycle_cholesky_three_solves(Native, spd, Bspd)
    _, generic_solved = cycle_cholesky_three_solves(Generic, spd, Bspd)
    native_etas = map(X -> gate_solution(spd, X, Bspd, p), native_solved)
    generic_etas = map(X -> gate_solution(spd, X, Bspd, p), generic_solved)
    @assert all(
        scaled_close(native_solved[i], generic_solved[i], p; dimension = n)
        for i in eachindex(native_solved)
    )

    native_X, native_gram, native_rrqr =
        cycle_trsm_gram_rrqr(Native, L, rectangular, p)
    generic_X, generic_gram, generic_rrqr =
        cycle_trsm_gram_rrqr(Generic, L, rectangular, p)
    @assert BFLA.issuccess(native_rrqr) && BFLA.issuccess(generic_rrqr)
    @assert scaled_close(native_X, generic_X, p; dimension = n)
    @assert scaled_close(native_gram, generic_gram, p; dimension = n)
    @assert BFLA.factor_rank(native_rrqr) == BFLA.factor_rank(generic_rrqr)

    native_ldlt, native_Xldlt = cycle_ldlt_multi_rhs(Native, indefinite, Bindef)
    generic_ldlt, generic_Xldlt =
        cycle_ldlt_multi_rhs(Generic, indefinite, Bindef)
    @assert BFLA.issuccess(native_ldlt) && BFLA.issuccess(generic_ldlt)
    native_eta_ldlt = gate_solution(indefinite, native_Xldlt, Bindef, p)
    generic_eta_ldlt = gate_solution(indefinite, generic_Xldlt, Bindef, p)
    @assert scaled_close(native_Xldlt, generic_Xldlt, p; dimension = n)
    @assert BFLA.factor_inertia(native_ldlt) == BFLA.factor_inertia(generic_ldlt)

    gate_facts = (
        native = (
            solve_eta = maximum(native_etas),
            rrqr_rank = BFLA.factor_rank(native_rrqr),
            ldlt_eta = native_eta_ldlt,
            inertia = BFLA.factor_inertia(native_ldlt),
        ),
        generic = (
            solve_eta = maximum(generic_etas),
            rrqr_rank = BFLA.factor_rank(generic_rrqr),
            ldlt_eta = generic_eta_ldlt,
            inertia = BFLA.factor_inertia(generic_ldlt),
        ),
    )

    for (backend_name, backend, facts) in (
        ("native", Native, gate_facts.native),
        ("generic", Generic, gate_facts.generic),
    )
        syrk_workspace = BFLA.BFLAWorkspace(p; workers = 1)
        solve_workspace = BFLA.BFLAWorkspace(p; workers = 1)
        workspace_chol = cycle_syrk_cholesky(
            backend, source, p; workspace = syrk_workspace,
        )
        workspace_factor, workspace_solved = cycle_cholesky_three_solves(
            backend, spd, Bspd; workspace = solve_workspace,
        )
        @assert triangle_scaled_close(
            BFLA.factor_matrix(workspace_chol),
            BFLA.factor_matrix(
                backend === Native ? native_chol : generic_chol,
            ),
            p,
            Lower;
            dimension = n,
        )
        @assert BFLA.issuccess(workspace_factor)
        @assert all(
            scaled_close(
                workspace_solved[index],
                backend === Native ? native_solved[index] : generic_solved[index],
                p;
                dimension = n,
            )
            for index in eachindex(workspace_solved)
        )
        report_gate("syrk_cholesky", backend_name, p, n; parity=true)
        report_gate(
            "syrk_cholesky_workspace", backend_name, p, n; parity=true,
        )
        report_gate("cholesky_three_solves", backend_name, p, n;
                    max_backward_error=facts.solve_eta, parity=true)
        report_gate(
            "cholesky_three_solves_workspace",
            backend_name,
            p,
            n;
            max_backward_error=facts.solve_eta,
            parity=true,
        )
        report_gate("trsm_gram_rrqr", backend_name, p, n;
                    rank=facts.rrqr_rank, parity=true)
        report_gate("ldlt_multi_rhs", backend_name, p, n;
                    backward_error=facts.ldlt_eta,
                    inertia=facts.inertia, parity=true)

        for (workload, operation) in (
            ("syrk_cholesky", () -> cycle_syrk_cholesky(backend, source, p)),
            ("syrk_cholesky_workspace", () -> cycle_syrk_cholesky(
                backend, source, p; workspace = syrk_workspace,
            )),
            ("cholesky_three_solves", () ->
                cycle_cholesky_three_solves(backend, spd, Bspd)),
            ("cholesky_three_solves_workspace", () ->
                cycle_cholesky_three_solves(
                    backend, spd, Bspd; workspace = solve_workspace,
                )),
            ("trsm_gram_rrqr", () ->
                cycle_trsm_gram_rrqr(backend, L, rectangular, p)),
            ("ldlt_multi_rhs", () ->
                cycle_ldlt_multi_rhs(backend, indefinite, Bindef)),
        )
            result = measure(operation; warmup = WARMUP, samples = SAMPLES)
            report_measurement(
                workload = workload,
                backend = backend_name,
                precision = p,
                size = n,
                threads = cycle_effective_threads(backend, workload),
                block_size = backend === Native ? BLOCK_SIZE : nothing,
                result = result,
            )
        end
    end
end

println("=== BFLA production-cycle benchmark ===")
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
    benchmark_fixture(p, n)
end
println((final_max_rss_bytes = Int(Sys.maxrss()),))
println("=== done ===")
