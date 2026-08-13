using Statistics
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

using BigFloatLinearAlgebra:
    NoTrans,
    Trans,
    Lower,
    Upper,
    LeftSide,
    RightSide,
    UnitDiagonal,
    NonUnitDiagonal

function measure(
    f::Function;
    warmup::Int=2,
    samples::Int=10,
    setup::Function=() -> nothing,
)
    setup()
    cold_rss_before = Int(Sys.maxrss())
    cold = @timed f()
    cold_rss_after = Int(Sys.maxrss())
    for _ in 1:warmup
        setup()
        f()
    end
    times = Vector{Float64}(undef, samples)
    allocs = Vector{Int}(undef, samples)
    for s in 1:samples
        setup()
        GC.gc()
        t = @timed f()
        times[s] = t.time
        allocs[s] = t.bytes
    end
    sort!(times)
    sort!(allocs)
    allocated_bytes_median = round(Int, Statistics.median(allocs))
    return (
        cold = cold.time,
        cold_bytes = cold.bytes,
        cold_rss_delta = max(cold_rss_after - cold_rss_before, 0),
        median = Statistics.median(times),
        iqr = Statistics.quantile(times, 0.75) - Statistics.quantile(times, 0.25),
        min = times[1],
        max = times[end],
        allocs = allocated_bytes_median,
        allocated_bytes_median = allocated_bytes_median,
        rss_bytes = Int(Sys.maxrss()),
    )
end

function report_measurement(; workload, backend, precision, size, threads,
                            block_size=nothing, result)
    println(
        "workload=", workload,
        " backend=", backend,
        " precision=", precision,
        " size=", size,
        " threads=", threads,
        " block_size=", block_size === nothing ? "none" : block_size,
        " cold_s=", round(result.cold; sigdigits = 6),
        " cold_bytes=", result.cold_bytes,
        " cold_rss_delta_bytes=", result.cold_rss_delta,
        " warm_median_s=", round(result.median; sigdigits = 6),
        " warm_iqr_s=", round(result.iqr; sigdigits = 6),
        " warm_min_s=", round(result.min; sigdigits = 6),
        " warm_max_s=", round(result.max; sigdigits = 6),
        " warm_median_bytes=", result.allocated_bytes_median,
        " rss_bytes=", result.rss_bytes,
    )
end

function environment()
    source_commit = get(ENV, "BFLA_SOURCE_COMMIT", "")
    if isempty(source_commit)
        source_commit = try
            readchomp(`git rev-parse HEAD`)
        catch
            "unavailable"
        end
    end
    return (
        julia = string(VERSION),
        threads = Threads.nthreads(),
        cpu = string(Sys.CPU_NAME),
        max_rss_bytes = Int(Sys.maxrss()),
        source_commit = source_commit,
    )
end

function eps_bits(p::Int)
    value = BigFloat(0; precision = p)
    BFLA._mpfr_set_ui_2exp!(value, 1, 1 - p)
    return value
end

function benchmark_threshold(p::Int, dimension::Int; constant::Int=10_000)
    value = BigFloat(0; precision = p)
    BFLA.MA.operate_to!(
        value,
        *,
        BigFloat(constant * max(dimension, 1); precision = p),
        eps_bits(p),
    )
    return value
end

function max_abs_difference(A, B, p::Int)
    size(A) == size(B) || error("comparison shape mismatch")
    maximum_value = BigFloat(0; precision = p)
    difference = BigFloat(0; precision = p)
    @inbounds for index in eachindex(A, B)
        BFLA.MA.operate_to!(difference, -, A[index], B[index])
        signbit(difference) && BFLA.MA.operate!(-, difference)
        difference > maximum_value &&
            BFLA.MA.operate_to!(maximum_value, copy, difference)
    end
    return maximum_value
end

function scaled_close(A, B, p::Int; dimension::Int=max(size(A)...))
    difference = max_abs_difference(A, B, p)
    scale = max(
        BFLA.norminf(Native, A),
        BFLA.norminf(Native, B),
        BigFloat(1; precision = p),
    )
    bound = BigFloat(0; precision = p)
    BFLA.MA.operate_to!(
        bound, *, benchmark_threshold(p, dimension), scale,
    )
    return difference <= bound
end

function triangle_scaled_close(A, B, p::Int, triangle;
                               dimension::Int=size(A, 1))
    size(A) == size(B) || error("comparison shape mismatch")
    size(A, 1) == size(A, 2) || error("triangle comparison requires square data")
    maximum_difference = BigFloat(0; precision = p)
    maximum_scale = BigFloat(1; precision = p)
    difference = BigFloat(0; precision = p)
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        authoritative = triangle === Lower ? i >= j : i <= j
        authoritative || continue
        BFLA.MA.operate_to!(difference, -, A[i, j], B[i, j])
        signbit(difference) && BFLA.MA.operate!(-, difference)
        difference > maximum_difference &&
            BFLA.MA.operate_to!(maximum_difference, copy, difference)
        abs(A[i, j]) > maximum_scale &&
            BFLA.MA.operate_to!(maximum_scale, abs, A[i, j])
        abs(B[i, j]) > maximum_scale &&
            BFLA.MA.operate_to!(maximum_scale, abs, B[i, j])
    end
    bound = BigFloat(0; precision = p)
    BFLA.MA.operate_to!(
        bound, *, benchmark_threshold(p, dimension), maximum_scale,
    )
    return maximum_difference <= bound
end

function parse_int_tuple(name::AbstractString, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return Tuple(parse(Int, strip(value)) for value in split(raw, ','))
end

function owned_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    return A
end
