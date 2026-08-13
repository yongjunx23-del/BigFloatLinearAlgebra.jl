# Reusable ownership-safe scratch. Every mutable BigFloat scalar and buffer is
# independently owned, created at an explicit precision, and worker-local so
# concurrent tasks never share an MPFR accumulator.

"""
    BFLAWorkspace(precision_bits; workers = Threads.nthreads(), scalar_slots = 16)

Caller-managed, ownership-safe scratch storage. All BigFloat storage is created
at exactly `precision_bits` bits (never from ambient `setprecision`) and is
split per worker so concurrent callers can reserve disjoint MPFR objects.
Worker-local identity buffers support the explicit Cholesky ownership scan
without retaining references or pointers to matrix elements.

Use [`workspace_scratch!`](@ref) and [`workspace_buffer!`](@ref) to obtain
worker-local MPFR scratch. Cholesky can additionally consume a worker-local
identity buffer when explicitly passed `workspace=ws`; other kernels do not
consume workspace storage. The caller owns the lifetime, worker assignment,
and synchronization policy, and concurrent calls must use distinct workers.
"""
mutable struct BFLAWorkspace
    precision_bits::Int
    workers::Int
    scalar_slots::Int
    scalars::Vector{Vector{BigFloat}}
    buffers::Vector{Vector{BigFloat}}
    identity_buffers::Vector{Vector{UInt}}
end

function BFLAWorkspace(precision_bits::Int; workers::Int = Threads.nthreads(), scalar_slots::Int = 16)
    precision_bits > 0 || throw(ArgumentError("precision_bits must be positive"))
    workers >= 1 || throw(ArgumentError("workers must be at least 1"))
    scalar_slots >= 1 || throw(ArgumentError("scalar_slots must be at least 1"))
    return BFLAWorkspace(
        precision_bits,
        workers,
        scalar_slots,
        [BigFloat[] for _ in 1:workers],
        [BigFloat[] for _ in 1:workers],
        [UInt[] for _ in 1:workers],
    )
end

function _workspace_identity_buffer(
    ws::BFLAWorkspace,
    worker::Int,
    precision_bits::Int,
    operation::AbstractString,
)
    ws.precision_bits == precision_bits || throw(PrecisionMismatch(
        precision_bits, ws.precision_bits, nothing,
    ))
    1 <= worker <= ws.workers || throw(ArgumentError(
        "$operation: workspace_worker must be in 1:$(ws.workers)",
    ))
    return ws.identity_buffers[worker]
end

function _workspace_identity_buffer(
    ::Nothing,
    workspace_worker::Int,
    ::Int,
    operation::AbstractString,
)
    workspace_worker == 1 || throw(ArgumentError(
        "$operation: workspace_worker requires a workspace",
    ))
    return nothing
end

workspace_precision(ws::BFLAWorkspace) = ws.precision_bits
workspace_workers(ws::BFLAWorkspace) = ws.workers

"""
    workspace_scratch!(ws, worker, slot) -> BigFloat

Return (allocating on first use) a worker-local mutable `BigFloat` at the
workspace precision. `worker` must be in `1:workspace_workers(ws)`.
"""
function workspace_scratch!(ws::BFLAWorkspace, worker::Int, slot::Int)
    1 <= worker <= ws.workers || throw(ArgumentError("worker out of range"))
    1 <= slot <= ws.scalar_slots || throw(ArgumentError("slot out of range"))
    scalars = ws.scalars[worker]
    while length(scalars) < slot
        push!(scalars, BigFloat(0; precision = ws.precision_bits))
    end
    return scalars[slot]
end

"""
    workspace_buffer!(ws, worker, length) -> Vector{BigFloat}

Return (growing on first use) a worker-local, independently owned buffer of at
least `length` elements at the workspace precision. The returned storage is
zero-initialized the first time it is extended.
"""
function workspace_buffer!(ws::BFLAWorkspace, worker::Int, n::Int)
    1 <= worker <= ws.workers || throw(ArgumentError("worker out of range"))
    n >= 0 || throw(ArgumentError("length must be non-negative"))
    buffer = ws.buffers[worker]
    if n > length(buffer)
        old = length(buffer)
        resize!(buffer, n)
        for i in (old + 1):n
            buffer[i] = BigFloat(0; precision = ws.precision_bits)
        end
    end
    return buffer
end
