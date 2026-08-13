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

function measure(f::Function; warmup::Int=2, samples::Int=10)
    for _ in 1:warmup
        f()
    end
    times = Vector{Float64}(undef, samples)
    allocs = Vector{Int}(undef, samples)
    for s in 1:samples
        GC.gc()
        t = @timed f()
        times[s] = t.time
        allocs[s] = t.bytes
    end
    sort!(times)
    sort!(allocs)
    return (
        median = Statistics.median(times),
        iqr = Statistics.quantile(times, 0.75) - Statistics.quantile(times, 0.25),
        min = times[1],
        max = times[end],
        allocs = Statistics.median(allocs),
    )
end

function environment()
    return (
        julia = string(VERSION),
        threads = Threads.nthreads(),
        cpu = string(Sys.CPU_NAME),
        max_rss_bytes = Sys.maxrss(),
        source_commit = get(ENV, "BFLA_SOURCE_COMMIT", "not supplied"),
    )
end

eps_bits(p::Int) = BigFloat(
    BigFloat(2; precision = p)^(1 - p); precision = p,
)

function parse_int_tuple(name::AbstractString, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return Tuple(parse(Int, strip(value)) for value in split(raw, ','))
end

function owned_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(2 * rand(rng) - 1; precision = p)
    end
    return A
end
