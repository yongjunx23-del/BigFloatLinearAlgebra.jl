# Precision-specific reusable factor caches.
#
# A cache owns every mutable BigFloat destination it uses: the factor matrix,
# per-worker scalar accumulators, and the metadata arrays. `prepare!` is the only
# place storage is (re)allocated; changing the precision or size is an explicit
# `prepare!` act, never something that happens inside the hot loops. After
# warm-up, `factorize!`, `solve!`, residual and refinement paths write into the
# cache's existing BigFloat destinations rather than replacing elements, so
# object identity is preserved and no new Julia objects are allocated.
#
# Caches mirror the field names of the corresponding allocating factor structs so
# the solver-grade kernels (`_cholesky_solve!`, `_ldlt_solve_common!`,
# `_apply_q_common!`, `_qr_solve_common!`, `_lu_solve!`) operate on them
# unchanged. The cache lifecycle is *borrowed*: the caller owns its lifetime and
# worker/synchronization policy. The ordinary allocating factor API is untouched
# and remains independent.

"""
    AbstractFactorCache

Supertype for all reusable, precision-specific factor caches.
"""
abstract type AbstractFactorCache end

factor_backend(cache::AbstractFactorCache) = cache.backend
factor_precision(cache::AbstractFactorCache) = cache.precision_bits
factor_status(cache::AbstractFactorCache) = cache.status
issuccess(cache::AbstractFactorCache) = cache.status.kind === :success
factor_failure_position(cache::AbstractFactorCache) = cache.status.position

"""
    factor_prepared(cache) -> Bool

Whether `prepare!` has been called for the current size/precision.
"""
factor_prepared(cache::AbstractFactorCache) = cache.prepared

"""
    factor_size(cache) -> Union{Nothing,Int}

The order of the factor matrix reserved by `prepare!`, or `nothing` if not
prepared.
"""
function factor_size(cache::AbstractFactorCache)
    cache.prepared || return nothing
    return cache.n
end

"""
    invalidate!(cache) -> cache

Mark the cache as holding no valid factorization. The owned factor matrix,
metadata arrays, and reusable refinement scratch are all preserved for reuse;
only the status is reset. A subsequent `factorize!` overwrites the same owned
destinations and `refine_once!` reuses the retained refinement storage.
"""
function invalidate!(cache::AbstractFactorCache)
    cache.status = FactorStatus(:unprepared, nothing)
    return cache
end

# --- shared validation -----------------------------------------------------

function _cache_require_prepared(cache::AbstractFactorCache, op::AbstractString)
    cache.prepared || throw(ArgumentError(
        "$op: cache has not been prepared; call prepare!(cache, n, precision_bits)",
    ))
    return nothing
end

function _cache_require_success(cache::AbstractFactorCache, op::AbstractString)
    _cache_require_prepared(cache, op)
    issuccess(cache) || throw(ArgumentError(
        "$op: cache factor status is not successful ($(cache.status.kind))",
    ))
    return nothing
end

function _require_cache_matrix(cache, A::AbstractMatrix{BigFloat}, op::AbstractString)
    n = cache.n
    size(A) == (n, n) || throw(DimensionMismatch(
        "$op: coefficient matrix must be $(n)x$(n), got $(size(A))",
    ))
    p = _require_precision(_check_precision(A), op)
    p == cache.precision_bits || throw(PrecisionMismatch(
        cache.precision_bits, p, nothing,
    ))
    return nothing
end

function _require_cache_rhs(cache, b::AbstractVecOrMat{BigFloat}, op::AbstractString)
    size(b, 1) == cache.n || throw(DimensionMismatch(
        "$op: right-hand side rows ($(size(b, 1))) differ from cache order " *
        "$(cache.n)",
    ))
    p = _require_precision(_check_precision(b), op)
    p == cache.precision_bits || throw(PrecisionMismatch(
        cache.precision_bits, p, nothing,
    ))
    return nothing
end

# Write source values into existing destination objects (never replace).
function _cache_copy_into!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
    op::AbstractString,
)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "$op: destination and source axes differ",
    ))
    Base.mightalias(destination, source) && throw(ArgumentError(
        "$op: destination must not alias the source",
    ))
    p = _require_precision(_check_precision(source), op)
    dp = _require_precision(_check_precision(destination), op)
    p == dp || throw(PrecisionMismatch(dp, p, nothing))
    @inbounds for index in eachindex(destination, source)
        MA.operate_to!(destination[index], copy, source[index])
    end
    return destination
end

# Copy an independently-owned RHS into the solution destination. On the warm-up
# call the destination may carry a stale/ambient precision (e.g. LinearSolve's
# `similar`+`fill!` initialization); we then replace each element with a
# source-precision copy so the destination thereafter carries the factor
# precision. On every later call the destination already has the right precision,
# so we write into the existing owned objects without replacing elements — the
# zero-allocation hot path.
function _cache_rhs_into!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
    op::AbstractString,
)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "$op: solution and right-hand side axes differ",
    ))
    Base.mightalias(destination, source) && throw(ArgumentError(
        "$op: solution must not alias the right-hand side",
    ))
    p = _require_precision(_check_precision(source), op)
    dp = _require_precision(_check_precision(destination), op)
    if dp != p
        @inbounds for index in eachindex(destination, source)
            destination[index] = MA.mutable_copy(source[index])
        end
    else
        @inbounds for index in eachindex(destination, source)
            MA.operate_to!(destination[index], copy, source[index])
        end
    end
    return destination
end

# Precision-specific scalar accumulators owned by a cache. Holding these across
# calls is what makes repeated factorization and solve not allocate fresh
# BigFloat objects.
mutable struct CacheScalars
    precision_bits::Int
    zero::BigFloat
    one::BigFloat
    minus_one::BigFloat
    acc::BigFloat
    buffer::BigFloat
    difference::BigFloat
    product::BigFloat
    maximum_value::BigFloat
    candidate::BigFloat
    # LDLT normalized 2x2 pivot scratch
    norm_a::BigFloat
    norm_e1::BigFloat
    norm_e2::BigFloat
    norm_c::BigFloat
    norm_det::BigFloat
    norm_r1::BigFloat
    norm_r2::BigFloat
    norm_y1::BigFloat
    norm_y2::BigFloat
    norm_work::BigFloat
    a1::BigFloat
    a2::BigFloat
end

function CacheScalars(precision_bits::Int)
    mk(v) = BigFloat(v; precision = precision_bits)
    return CacheScalars(
        precision_bits,
        mk(0), mk(1), mk(-1),
        mk(0), mk(0), mk(0), mk(0),
        mk(0), mk(0),
        mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0), mk(0),
    )
end

# Owned refinement scratch. Allocated by prepare_refinement! so the refinement
# step itself writes into existing destinations.
mutable struct CacheRefineStorage
    residual::AbstractArray{BigFloat}
    correction::AbstractArray{BigFloat}
    precision_bits::Int
end

"""
    prepare_refinement!(cache, nrhs=1) -> cache
    prepare_refinement!(cache, rhs_template) -> cache

Eagerly allocate the cache's owned residual/correction scratch so a subsequent
`refine_once!` writes into existing destinations. The integer form reserves
`n × nrhs` (matrix RHS); the array form reserves the exact shape of
`rhs_template` (a `Vector{BigFloat}` or `Matrix{BigFloat}`), so a Vector RHS and
an `n×1` Matrix RHS are kept distinct and no `refine_once!` reallocates for the
shape it was prepared for. It is also called by `prepare!` (with `nrhs`).
"""
function prepare_refinement!(cache::AbstractFactorCache, nrhs::Int=1)
    nrhs >= 1 || throw(ArgumentError("prepare_refinement!: nrhs must be positive"))
    n = cache.n
    cache.refine = CacheRefineStorage(
        owned_zeros(BigFloat, n, nrhs; precision_bits = cache.precision_bits),
        owned_zeros(BigFloat, n, nrhs; precision_bits = cache.precision_bits),
        cache.precision_bits,
    )
    return cache
end

function prepare_refinement!(cache::AbstractFactorCache, rhs_template::AbstractArray{BigFloat})
    p = _require_precision(_check_precision(rhs_template), "prepare_refinement!")
    p == cache.precision_bits || throw(PrecisionMismatch(cache.precision_bits, p, nothing))
    size(rhs_template, 1) == cache.n || throw(DimensionMismatch(
        "prepare_refinement!: RHS rows ($(size(rhs_template, 1))) differ from " *
        "cache order $(cache.n)",
    ))
    cache.refine = CacheRefineStorage(
        owned_zeros(BigFloat, size(rhs_template)...; precision_bits = cache.precision_bits),
        owned_zeros(BigFloat, size(rhs_template)...; precision_bits = cache.precision_bits),
        cache.precision_bits,
    )
    return cache
end


# --- Cholesky --------------------------------------------------------------

"""
    BFLACholeskyCache(backend; triangle=Lower)

Reusable Cholesky cache. The factor matrix and scalar scratch are owned by the
cache at an explicit precision. Repeated `factorize!` overwrites the same owned
matrix; repeated `solve!` writes into existing destinations.
"""
mutable struct BFLACholeskyCache{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractFactorCache
    factors::M
    backend::B
    triangle::Triangle
    precision_bits::Int
    status::FactorStatus
    prepared::Bool
    n::Int
    scalars::Union{Nothing,CacheScalars}
    refine::Union{Nothing,CacheRefineStorage}
    workspace::Union{Nothing,BFLAWorkspace}
end

function BFLACholeskyCache(backend::AbstractBFLABackend; triangle::Triangle=Lower)
    return BFLACholeskyCache{Matrix{BigFloat},typeof(backend)}(
        Matrix{BigFloat}(undef, 0, 0), backend, triangle, 0,
        FactorStatus(:unprepared, nothing), false, 0, nothing, nothing, nothing,
    )
end

factor_kind(::BFLACholeskyCache) = :cholesky
factor_triangle(cache::BFLACholeskyCache) = cache.triangle
factor_matrix(cache::BFLACholeskyCache) = cache.factors

"""
    prepare!(cache::BFLACholeskyCache, n, precision_bits; nrhs=1,
             workspace_workers=1)

Reserve `n × n` factor storage and scalar/workspace scratch at
`precision_bits`. The only allocating entry point.
"""
function prepare!(
    cache::BFLACholeskyCache,
    n::Int,
    precision_bits::Int;
    nrhs::Int=1,
    workspace_workers::Int=1,
)
    n >= 0 || throw(ArgumentError("prepare!: n must be non-negative"))
    precision_bits > 0 || throw(ArgumentError("prepare!: precision_bits must be positive"))
    nrhs >= 1 || throw(ArgumentError("prepare!: nrhs must be positive"))
    workspace_workers >= 1 || throw(ArgumentError("prepare!: workspace_workers must be positive"))
    cache.factors = owned_zeros(BigFloat, n, n; precision_bits = precision_bits)
    cache.precision_bits = precision_bits
    cache.n = n
    cache.scalars = CacheScalars(precision_bits)
    cache.workspace = BFLAWorkspace(precision_bits; workers = workspace_workers)
    cache.status = FactorStatus(:unprepared, nothing)
    prepare_refinement!(cache, nrhs)
    cache.prepared = true
    return cache
end

"""
    factorize!(cache::BFLACholeskyCache, A) -> cache

Deep-copy `A` into the cache's owned factor matrix at the cache precision and
factor it in place. The authoritative triangle is read and written. A
non-positive-definite or non-finite matrix yields a non-success status rather
than throwing.
"""
function factorize!(
    cache::BFLACholeskyCache,
    A::AbstractMatrix{BigFloat},
)
    _cache_require_prepared(cache, "factorize!")
    _require_cache_matrix(cache, A, "factorize!")
    triangle = cache.triangle
    _require_valid_triangle(triangle, "factorize!")
    _cache_copy_into!(cache.factors, A, "factorize!")
    factors = cache.factors
    if !_triangle_finite(factors, triangle)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    info = _cache_cholesky_dispatch!(
        cache.backend, factors, triangle, cache.precision_bits, cache.scalars,
    )
    if !_triangle_finite(factors, triangle)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    cache.status = info == 0 ? SUCCESS_STATUS :
        FactorStatus(:not_positive_definite, info)
    return cache
end

# Dispatch Cholesky into cache-owned scalar scratch. The Native path is
# zero-allocation; Generic is the reference backend and may allocate.
@inline function _cache_cholesky_dispatch!(
    backend::NativeBackend,
    A::AbstractMatrix{BigFloat},
    triangle::Triangle,
    p::Int,
    scalars::Union{Nothing,CacheScalars},
)
    triangle === Lower ||
        _unsupported(NativeBackend(), :cholesky, "NativeBackend supports triangle=Lower only in phase 1")
    return _cache_cholesky_native!(A, p, scalars)
end
@inline function _cache_cholesky_dispatch!(
    backend::GenericBackend,
    A::AbstractMatrix{BigFloat},
    triangle::Triangle,
    p::Int,
    ::Union{Nothing,CacheScalars},
)
    return _cholesky_dispatch!(backend, A, triangle, p, KernelConfig())
end

# Zero-allocation Native Cholesky into the cache's owned matrix and scalar
# scratch. Mirror of `_cholesky!` using caller-owned scalars.
function _cache_cholesky_native!(A::AbstractMatrix{BigFloat}, p::Int, s::CacheScalars)
    acc = s.acc
    buffer = s.buffer
    difference = s.difference
    k = size(A, 1)
    k == 0 && return 0
    @inbounds for j in 1:k
        if j > 1
            MA.operate!(zero, acc)
            for index in 1:(j - 1)
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[j, index], A[j, index])
            end
            MA.operate_to!(difference, -, A[j, j], acc)
            djj = difference
        else
            djj = A[j, j]
        end
        djj <= 0 && return j
        _mpfr_sqrt!(A[j, j], djj)
        Ljj = A[j, j]
        for i in (j + 1):k
            if j > 1
                MA.operate!(zero, acc)
                for index in 1:(j - 1)
                    MA.buffered_operate!(
                        buffer, MA.add_mul, acc, A[i, index], A[j, index],
                    )
                end
                MA.operate_to!(difference, -, A[i, j], acc)
                numerator = difference
            else
                numerator = A[i, j]
            end
            _mpfr_div!(A[i, j], numerator, Ljj)
        end
    end
    return 0
end

"""
    solve!(x, cache, b) -> x

Write the solution of the factored system into `x` (existing owned elements).
For a lower factor this solves `(L*Lᵀ)x = b`; an upper factor (GenericBackend)
solves `(Uᵀ*U)x = b`.
"""
function solve!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLACholeskyCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_checked_solve_check!(x, cache, b, "solve!")
    _cache_rhs_repair!(x, b, "solve!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _cholesky_solve!(
        cache.backend, cache.factors, cache.triangle, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

"""
    solve_trusted!(x, cache, b) -> x

Trusted, ownership-safe solve for the solver-facing hot path. `x` must already
be independently-owned at the factor precision (e.g. from `owned_zeros`); BFLA
copies `b` into the existing `x` objects and solves in place, allocating no new
BigFloat objects. Unlike the checked [`solve!`](@ref), this skips the ownership
and precision scan, which the caller guarantees.
"""
function solve_trusted!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLACholeskyCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_trusted_solve_check!(x, cache, b, "solve_trusted!")
    _cache_rhs_write!(x, b, "solve_trusted!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _cholesky_solve!(
        cache.backend, cache.factors, cache.triangle, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

"""
    solve!(cache, b) -> Vector or Matrix

Allocating convenience: copy `b` into a fresh owned solution and solve it.
Prefer the three-argument form to avoid the allocation on the hot path.
"""
function solve(
    cache::BFLACholeskyCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_require_success(cache, "solve")
    x = owned_zeros(BigFloat, size(b)...; precision_bits = cache.precision_bits)
    return solve_trusted!(x, cache, b)
end

# Select the embedded workspace for a cache solve. The cache owns a worker-local
# BFLAWorkspace; worker 1 is used for the borrowed (single-threaded) contract.
@inline function _cache_solve_workspace(cache::AbstractFactorCache, worker::Int)
    ws = cache.workspace
    ws === nothing && return nothing, worker
    1 <= worker <= ws.workers || throw(ArgumentError(
        "solve!: workspace_worker must be in 1:$(ws.workers)",
    ))
    return ws, worker
end

# --- checked / trusted solve contract --------------------------------------

# Checked solve: replace every solution element with an independently-owned copy
# of the RHS at the RHS precision, repairing both stale precision and shared
# ownership. Safe for arbitrary destinations (e.g. `fill(BigFloat(...), n)` or
# LinearSolve-style `similar`+`fill!`); allocates by design.
function _cache_rhs_repair!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
    op::AbstractString,
)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "$op: solution and right-hand side axes differ",
    ))
    Base.mightalias(destination, source) && throw(ArgumentError(
        "$op: solution must not alias the right-hand side",
    ))
    _require_precision(_check_precision(source), op)
    @inbounds for index in eachindex(destination, source)
        destination[index] = MA.mutable_copy(source[index])
    end
    return destination
end

# Trusted solve: copy the RHS into an already-owned, precision-consistent
# destination by writing into the existing BigFloat objects (no replacement, no
# ownership scan). The caller must guarantee the destination is independently
# owned at the factor precision (e.g. from `owned_zeros`).
function _cache_rhs_write!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
    op::AbstractString,
)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "$op: solution and right-hand side axes differ",
    ))
    Base.mightalias(destination, source) && throw(ArgumentError(
        "$op: solution must not alias the right-hand side",
    ))
    p = _require_precision(_check_precision(source), op)
    dp = _require_precision(_check_precision(destination), op)
    p == dp || throw(PrecisionMismatch(dp, p, nothing))
    @inbounds for index in eachindex(destination, source)
        MA.operate_to!(destination[index], copy, source[index])
    end
    return destination
end

# Shared validation for the checked `solve!`: status, dimensions, RHS precision,
# and aliasing of the solution against both the factor and the RHS.
function _cache_checked_solve_check!(x, cache, b, op)
    _cache_require_success(cache, op)
    (size(x, 1) == cache.n && size(b, 1) == cache.n) || throw(DimensionMismatch(
        "$op: cache, solution and right-hand side dimensions differ",
    ))
    _require_cache_rhs(cache, b, op)
    _require_no_alias(x, cache.factors, op)
    Base.mightalias(x, b) && throw(ArgumentError(
        "$op: solution must not alias the right-hand side",
    ))
    return nothing
end

# Minimal validation for the trusted `solve_trusted!`: status, dimensions, and
# aliasing against the factor and the RHS. The caller still guarantees the
# destination's independent BigFloat ownership and precision.
function _cache_trusted_solve_check!(x, cache, b, op)
    _cache_require_success(cache, op)
    (size(x, 1) == cache.n && size(b, 1) == cache.n) || throw(DimensionMismatch(
        "$op: cache, solution and right-hand side dimensions differ",
    ))
    _require_no_alias(x, cache.factors, op)
    Base.mightalias(x, b) && throw(ArgumentError(
        "$op: solution must not alias the right-hand side",
    ))
    return nothing
end

# --- LU --------------------------------------------------------------------

"""
    BFLALUCache(backend)

Reusable partial-pivoting LU cache. Owned factor matrix, pivots, permutation,
and scalar scratch at an explicit precision.
"""
mutable struct BFLALUCache{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractFactorCache
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    pivots::Vector{Int}
    perm::Vector{Int}
    prepared::Bool
    n::Int
    scalars::Union{Nothing,CacheScalars}
    refine::Union{Nothing,CacheRefineStorage}
    workspace::Union{Nothing,BFLAWorkspace}
end

function BFLALUCache(backend::AbstractBFLABackend)
    return BFLALUCache{Matrix{BigFloat},typeof(backend)}(
        Matrix{BigFloat}(undef, 0, 0), backend, 0,
        FactorStatus(:unprepared, nothing), Int[], Int[],
        false, 0, nothing, nothing, nothing,
    )
end

factor_kind(::BFLALUCache) = :lu
factor_matrix(cache::BFLALUCache) = cache.factors
factor_pivots(cache::BFLALUCache) = copy(cache.pivots)
function factor_perm(cache::BFLALUCache)
    _cache_require_success(cache, "factor_perm")
    return copy(cache.perm)
end

function prepare!(
    cache::BFLALUCache,
    n::Int,
    precision_bits::Int;
    nrhs::Int=1,
    workspace_workers::Int=1,
)
    n >= 0 || throw(ArgumentError("prepare!: n must be non-negative"))
    precision_bits > 0 || throw(ArgumentError("prepare!: precision_bits must be positive"))
    cache.factors = owned_zeros(BigFloat, n, n; precision_bits = precision_bits)
    cache.precision_bits = precision_bits
    cache.n = n
    cache.pivots = collect(1:n)
    cache.perm = collect(1:n)
    cache.scalars = CacheScalars(precision_bits)
    cache.workspace = BFLAWorkspace(precision_bits; workers = workspace_workers)
    cache.status = FactorStatus(:unprepared, nothing)
    prepare_refinement!(cache, nrhs)
    cache.prepared = true
    return cache
end

"""
    factorize!(cache::BFLALUCache, A) -> cache

Deep-copy `A` into the cache's owned factor matrix and factor it in place with
partial row pivoting. Singular or non-finite input yields a non-success status
rather than throwing.
"""
function factorize!(
    cache::BFLALUCache,
    A::AbstractMatrix{BigFloat},
)
    _cache_require_prepared(cache, "factorize!")
    _require_cache_matrix(cache, A, "factorize!")
    _cache_copy_into!(cache.factors, A, "factorize!")
    factors = cache.factors
    if !_all_finite(factors)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    info = _cache_lu!(cache)
    if !_all_finite(factors)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    _lu_rebuild_perm!(cache.perm, cache.pivots)
    cache.status = info == 0 ? SUCCESS_STATUS : FactorStatus(:singular, info)
    return cache
end

# Rebuild the final row permutation `perm` in place from the step pivots, so
# `factor_perm(cache)` and `factor_diagnostics(cache).permutation` report the
# permutation `A[perm, :] = L*U` consistent with the allocating LU factor.
function _lu_rebuild_perm!(perm::Vector{Int}, pivots::Vector{Int})
    n = length(pivots)
    @inbounds for i in 1:n
        perm[i] = i
    end
    @inbounds for k in 1:n
        pivot = pivots[k]
        perm[k], perm[pivot] = perm[pivot], perm[k]
    end
    return perm
end

# Native LU kernel: writes pivots into the cache's preallocated array and
# factorizes the cache's owned matrix in place, using cache-owned scalars
# (zero Julia allocation). Dispatched on the backend type so a Native cache can
# never execute a Generic kernel and vice versa.
function _cache_lu!(cache::BFLALUCache{<:Any,NativeBackend})
    n = cache.n
    factors = cache.factors
    pivots = cache.pivots
    scalars = cache.scalars
    @inbounds for i in 1:n
        pivots[i] = i
    end
    maximum_value = scalars.maximum_value
    candidate = scalars.candidate
    product = scalars.product
    @inbounds for k in 1:n
        pivot = k
        _lu_abs_to!(maximum_value, factors[k, k])
        for i in (k + 1):n
            _lu_abs_to!(candidate, factors[i, k])
            if candidate > maximum_value
                MA.operate_to!(maximum_value, copy, candidate)
                pivot = i
            end
        end
        pivots[k] = pivot
        iszero(maximum_value) && return k
        if pivot != k
            for j in 1:n
                factors[k, j], factors[pivot, j] = factors[pivot, j], factors[k, j]
            end
        end
        for i in (k + 1):n
            _mpfr_div!(factors[i, k], factors[i, k], factors[k, k])
            for j in (k + 1):n
                MA.operate_to!(product, *, factors[i, k], factors[k, j])
                MA.operate!(-, factors[i, j], product)
            end
        end
    end
    return 0
end

# Generic reference LU kernel. It delegates to the reference `_lu!` path, which
# may allocate its pivot metadata and scratch; this is the documented, honest
# behaviour of the Generic backend (it is a reference, not a zero-allocation
# target). It still writes pivots into the cache's owned array and reuses the
# cache's factor matrix.
function _cache_lu!(cache::BFLALUCache{<:Any,GenericBackend})
    n = cache.n
    p = cache.precision_bits
    info, pivots = _lu!(GenericBackend(), cache.factors, p)
    @inbounds for i in 1:n
        cache.pivots[i] = i <= length(pivots) ? pivots[i] : i
    end
    return info
end

function solve!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLALUCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_checked_solve_check!(x, cache, b, "solve!")
    _cache_rhs_repair!(x, b, "solve!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _lu_solve!(
        cache.backend, cache.factors, cache.pivots, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

"""
    solve_trusted!(x, cache, b) -> x

Trusted, ownership-safe solve for the solver-facing hot path. `x` must already
be independently-owned at the factor precision (e.g. from `owned_zeros`); BFLA
copies `b` into the existing `x` objects and solves in place, allocating no new
BigFloat objects. Unlike the checked [`solve!`](@ref), this skips the ownership
and precision scan, which the caller guarantees.
"""
function solve_trusted!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLALUCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_trusted_solve_check!(x, cache, b, "solve_trusted!")
    _cache_rhs_write!(x, b, "solve_trusted!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _lu_solve!(
        cache.backend, cache.factors, cache.pivots, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

function solve(
    cache::BFLALUCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_require_success(cache, "solve")
    x = owned_zeros(BigFloat, size(b)...; precision_bits = cache.precision_bits)
    return solve_trusted!(x, cache, b)
end

# --- LDLT (Bunch-Kaufman) --------------------------------------------------

"""
    BFLALDLTCache(backend)

Reusable symmetric-indefinite `P A Pᵀ = L D Lᵀ` cache. Owned factor matrix,
permutation, pivot-block sizes, subdiagonal-D flags, and scalar scratch at an
explicit precision.
"""
mutable struct BFLALDLTCache{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractFactorCache
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    perm::Vector{Int}
    blocks::Vector{Int}
    subdiag_is_d::Vector{Bool}
    prepared::Bool
    n::Int
    scalars::Union{Nothing,CacheScalars}
    refine::Union{Nothing,CacheRefineStorage}
    workspace::Union{Nothing,BFLAWorkspace}
end

function BFLALDLTCache(backend::AbstractBFLABackend)
    return BFLALDLTCache{Matrix{BigFloat},typeof(backend)}(
        Matrix{BigFloat}(undef, 0, 0), backend, 0,
        FactorStatus(:unprepared, nothing), Int[], Int[], Bool[],
        false, 0, nothing, nothing, nothing,
    )
end

factor_kind(::BFLALDLTCache) = :ldlt
factor_triangle(::BFLALDLTCache) = Lower
factor_matrix(cache::BFLALDLTCache) = cache.factors
function factor_perm(cache::BFLALDLTCache)
    _cache_require_success(cache, "factor_perm")
    return copy(cache.perm)
end
function factor_blocks(cache::BFLALDLTCache)
    _cache_require_success(cache, "factor_blocks")
    return copy(cache.blocks)
end

"""
    factor_inertia(cache::BFLALDLTCache) -> (npos, nneg, nzero)

Sylvester inertia of the factorized matrix from the `D` block diagonal. Only
valid after a successful `factorize!`.
"""
function factor_inertia(cache::BFLALDLTCache)
    _cache_require_success(cache, "factor_inertia")
    return _factor_inertia_unchecked(cache)
end

function prepare!(
    cache::BFLALDLTCache,
    n::Int,
    precision_bits::Int;
    nrhs::Int=1,
    workspace_workers::Int=1,
)
    n >= 0 || throw(ArgumentError("prepare!: n must be non-negative"))
    precision_bits > 0 || throw(ArgumentError("prepare!: precision_bits must be positive"))
    cache.factors = owned_zeros(BigFloat, n, n; precision_bits = precision_bits)
    cache.precision_bits = precision_bits
    cache.n = n
    cache.perm = collect(1:n)
    cache.blocks = Int[]
    cache.subdiag_is_d = falses(n)
    cache.scalars = CacheScalars(precision_bits)
    cache.workspace = BFLAWorkspace(precision_bits; workers = workspace_workers)
    cache.status = FactorStatus(:unprepared, nothing)
    prepare_refinement!(cache, nrhs)
    cache.prepared = true
    return cache
end

"""
    factorize!(cache::BFLALDLTCache, A) -> cache

Deep-copy `A` into the cache's owned factor matrix and factor it with
Bunch-Kaufman pivoting. Singular or non-finite input yields a non-success
status rather than throwing.
"""
function factorize!(
    cache::BFLALDLTCache,
    A::AbstractMatrix{BigFloat},
)
    _cache_require_prepared(cache, "factorize!")
    _require_cache_matrix(cache, A, "factorize!")
    _cache_copy_into!(cache.factors, A, "factorize!")
    factors = cache.factors
    if !_triangle_finite(factors, Lower)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    info = _cache_ldlt!(cache)
    if !_triangle_finite(factors, Lower)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    cache.status = info == 0 ? SUCCESS_STATUS :
        FactorStatus(:pivot_failure, info)
    return cache
end

# Bunch-Kaufman factorize reuses the allocating kernel for correctness; it
# writes perm/blocks into the cache's owned arrays. The kernel may allocate its
# small metadata; the cache solve/residual/refinement paths are zero-alloc.
function _cache_ldlt!(cache::BFLALDLTCache)
    info, perm, blocks = _ldlt!(cache.backend, cache.factors, cache.precision_bits)
    n = cache.n
    cache.perm = perm
    cache.blocks = blocks
    subdiag = cache.subdiag_is_d
    derived = _subdiag_is_d(blocks, n)
    @inbounds for i in 1:n
        subdiag[i] = i <= length(derived) && derived[i]
    end
    return info
end

function solve!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLALDLTCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_checked_solve_check!(x, cache, b, "solve!")
    _cache_rhs_repair!(x, b, "solve!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _ldlt_solve!(cache.backend, cache, x, workspace, worker)
    return x
end

"""
    solve_trusted!(x, cache, b) -> x

Trusted, ownership-safe solve for the solver-facing hot path. `x` must already
be independently-owned at the factor precision (e.g. from `owned_zeros`); BFLA
copies `b` into the existing `x` objects and solves in place, allocating no new
BigFloat objects. Unlike the checked [`solve!`](@ref), this skips the ownership
and precision scan, which the caller guarantees.
"""
function solve_trusted!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLALDLTCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_trusted_solve_check!(x, cache, b, "solve_trusted!")
    _cache_rhs_write!(x, b, "solve_trusted!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _ldlt_solve!(cache.backend, cache, x, workspace, worker)
    return x
end

function solve(
    cache::BFLALDLTCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_require_success(cache, "solve")
    x = owned_zeros(BigFloat, size(b)...; precision_bits = cache.precision_bits)
    return solve_trusted!(x, cache, b)
end

# --- RRQR ----------------------------------------------------------------

"""
    BFLARRQRCache(backend)

Reusable rank-revealing column-pivoted QR cache. Owned factor matrix,
Householder `tau`, column permutation `jpvt`, rank-policy scalars, and scratch
at an explicit precision.

**Scope (experimental):** this cache is currently *square-only* (`n × n`). It
does **not** yet support the rectangular / overdetermined systems that the
ordinary allocating [`qr`](@ref) handles. `tau` has length `n` and `jpvt` length
`n`. Do not present this cache as a full RRQR backend for rectangular inputs;
use the allocating `qr!` for those. Factorization may still allocate its pivot/
`tau` metadata (the solve path is zero-allocation).
"""
mutable struct BFLARRQRCache{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractFactorCache
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    tau::Vector{BigFloat}
    jpvt::Vector{Int}
    rank::Int
    tolerance::BigFloat
    atol::BigFloat
    rtol::BigFloat
    reference_scale::BigFloat
    effective_threshold::BigFloat
    prepared::Bool
    n::Int
    scalars::Union{Nothing,CacheScalars}
    refine::Union{Nothing,CacheRefineStorage}
    workspace::Union{Nothing,BFLAWorkspace}
end

function BFLARRQRCache(backend::AbstractBFLABackend)
    return BFLARRQRCache{Matrix{BigFloat},typeof(backend)}(
        Matrix{BigFloat}(undef, 0, 0), backend, 0,
        FactorStatus(:unprepared, nothing), BigFloat[], Int[], 0,
        BigFloat(0), BigFloat(0),
        BigFloat(0), BigFloat(0),
        BigFloat(0),
        false, 0, nothing, nothing, nothing,
    )
end

factor_kind(::BFLARRQRCache) = :rrqr
factor_matrix(cache::BFLARRQRCache) = cache.factors
function factor_jpvt(cache::BFLARRQRCache)
    _cache_require_success(cache, "factor_jpvt")
    return copy(cache.jpvt)
end
function factor_rank(cache::BFLARRQRCache)
    _cache_require_success(cache, "factor_rank")
    return cache.rank
end

function factor_Rdiag(cache::BFLARRQRCache)
    _cache_require_success(cache, "factor_Rdiag")
    n = cache.n
    Rdiag = BigFloat[]
    p = cache.precision_bits
    @inbounds for i in 1:min(n, length(cache.tau))
        push!(Rdiag, BigFloat(cache.factors[i, i]; precision = p))
    end
    return Rdiag
end

factor_rank_atol(cache::BFLARRQRCache) = MA.mutable_copy(cache.atol)
factor_rank_rtol(cache::BFLARRQRCache) = MA.mutable_copy(cache.rtol)
factor_rank_scale(cache::BFLARRQRCache) = MA.mutable_copy(cache.reference_scale)
factor_rank_threshold(cache::BFLARRQRCache) = MA.mutable_copy(cache.effective_threshold)

function prepare!(
    cache::BFLARRQRCache,
    n::Int,
    precision_bits::Int;
    nrhs::Int=1,
    workspace_workers::Int=1,
)
    n >= 0 || throw(ArgumentError("prepare!: n must be non-negative"))
    precision_bits > 0 || throw(ArgumentError("prepare!: precision_bits must be positive"))
    cache.factors = owned_zeros(BigFloat, n, n; precision_bits = precision_bits)
    cache.precision_bits = precision_bits
    cache.n = n
    cache.tau = BigFloat[]
    cache.jpvt = collect(1:n)
    cache.rank = 0
    z = BigFloat(0; precision = precision_bits)
    cache.tolerance = BigFloat(0; precision = precision_bits)
    cache.atol = BigFloat(0; precision = precision_bits)
    cache.rtol = BigFloat(0; precision = precision_bits)
    cache.reference_scale = BigFloat(0; precision = precision_bits)
    cache.effective_threshold = BigFloat(0; precision = precision_bits)
    cache.scalars = CacheScalars(precision_bits)
    cache.workspace = BFLAWorkspace(precision_bits; workers = workspace_workers)
    cache.status = FactorStatus(:unprepared, nothing)
    prepare_refinement!(cache, nrhs)
    cache.prepared = true
    return cache
end

"""
    factorize!(cache::BFLARRQRCache, A; tol=nothing, atol=nothing, rtol=nothing) -> cache

Deep-copy `A` into the cache's owned factor matrix and factor it with
column-pivoted Householder QR. Rank is derived from the explicit rank policy
(`atol`, `rtol`, or legacy `tol`). Non-finite input yields a non-success status.
"""
function factorize!(
    cache::BFLARRQRCache,
    A::AbstractMatrix{BigFloat};
    tol::Union{Nothing,BigFloat}=nothing,
    atol::Union{Nothing,BigFloat}=nothing,
    rtol::Union{Nothing,BigFloat}=nothing,
)
    _cache_require_prepared(cache, "factorize!")
    _require_cache_matrix(cache, A, "factorize!")
    _cache_copy_into!(cache.factors, A, "factorize!")
    p = cache.precision_bits
    factors = cache.factors
    if !_all_finite(factors)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    absolute, relative, scale, threshold = _qr_rank_policy(
        A, p, tol, atol, rtol, "factorize!",
    )
    zero_threshold = BigFloat(0; precision = p)
    tau, jpvt, _ = _qr!(cache.backend, factors, p, zero_threshold)
    if !_all_finite(factors) || !_all_finite(tau)
        cache.status = FactorStatus(:nonfinite, nothing)
        return cache
    end
    rank = _qr_rank_from_factors(factors, threshold)
    cache.tau = tau
    cache.jpvt = jpvt
    cache.rank = rank
    cache.tolerance = MA.mutable_copy(absolute)
    cache.atol = MA.mutable_copy(absolute)
    cache.rtol = MA.mutable_copy(relative)
    cache.reference_scale = MA.mutable_copy(scale)
    cache.effective_threshold = MA.mutable_copy(threshold)
    cache.status = SUCCESS_STATUS
    return cache
end

function solve!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLARRQRCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_checked_solve_check!(x, cache, b, "solve!")
    _cache_rhs_repair!(x, b, "solve!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _qr_solve!(cache.backend, cache, x, workspace, worker)
    return x
end

"""
    solve_trusted!(x, cache, b) -> x

Trusted, ownership-safe solve for the solver-facing hot path. `x` must already
be independently-owned at the factor precision (e.g. from `owned_zeros`); BFLA
copies `b` into the existing `x` objects and solves in place, allocating no new
BigFloat objects. Unlike the checked [`solve!`](@ref), this skips the ownership
and precision scan, which the caller guarantees.
"""
function solve_trusted!(
    x::AbstractVecOrMat{BigFloat},
    cache::BFLARRQRCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_trusted_solve_check!(x, cache, b, "solve_trusted!")
    _cache_rhs_write!(x, b, "solve_trusted!")
    workspace, worker = _cache_solve_workspace(cache, 1)
    _qr_solve!(cache.backend, cache, x, workspace, worker)
    return x
end

function solve(
    cache::BFLARRQRCache,
    b::AbstractVecOrMat{BigFloat},
)
    _cache_require_success(cache, "solve")
    x = owned_zeros(BigFloat, size(b)...; precision_bits = cache.precision_bits)
    return solve_trusted!(x, cache, b)
end

# --- factor_diagnostics ---------------------------------------------------

function factor_diagnostics(cache::BFLACholeskyCache)
    return (
        factor_kind = factor_kind(cache),
        triangle = factor_triangle(cache),
        failure_position = factor_failure_position(cache),
    )
end

function factor_diagnostics(cache::BFLALUCache)
    _cache_require_success(cache, "factor_diagnostics")
    return (
        factor_kind = factor_kind(cache),
        row_swap_count = count(k -> cache.pivots[k] != k, eachindex(cache.pivots)),
        permutation = copy(cache.perm),
        failure_position = factor_failure_position(cache),
    )
end

function factor_diagnostics(cache::BFLALDLTCache)
    _cache_require_success(cache, "factor_diagnostics")
    inertia = _factor_inertia_unchecked(cache)
    return (
        factor_kind = factor_kind(cache),
        inertia = inertia,
        permutation = copy(cache.perm),
        blocks = copy(cache.blocks),
        failure_position = factor_failure_position(cache),
    )
end

function factor_diagnostics(cache::BFLARRQRCache)
    _cache_require_success(cache, "factor_diagnostics")
    return (
        factor_kind = factor_kind(cache),
        rank = cache.rank,
        permutation = copy(cache.jpvt),
        failure_position = factor_failure_position(cache),
    )
end

# --- refinement -----------------------------------------------------------

"""
    refine_once!(cache, A, x, b) -> NamedTuple

Perform exactly one iterative-refinement correction on the cache's factor:
`residual = b - A*x`, solve the factor system for a correction, and update
`x += correction`. The cache owns the residual and correction scratch
(allocated by `prepare!` / `prepare_refinement!`), so once that storage is ready
the step writes into existing destinations. This is a single step; BFLA does not
iterate, choose a tolerance, raise precision, or switch backend.

Cache refinement is *factor-precision-only*: the residual, correction, and
backward error all use the cache's explicit precision. There is intentionally no
`residual_precision` keyword here; higher-precision residual refinement belongs
to the allocating-factor API and is not part of the cache contract.
"""
function refine_once!(
    cache::Union{BFLACholeskyCache,BFLALUCache,BFLALDLTCache,BFLARRQRCache},
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
)
    _cache_require_success(cache, "refine_once!")
    _require_cache_matrix(cache, A, "refine_once!")
    p = cache.precision_bits
    size(x) == size(b) || throw(DimensionMismatch(
        "refine_once!: solution and right-hand side dimensions differ",
    ))
    n = cache.n
    size(x, 1) == n || throw(DimensionMismatch(
        "refine_once!: cache and solution rows differ",
    ))
    residual = _cache_refine_storage(cache, size(b))
    return _refine_once_cached(cache, A, x, b, residual, p)
end

# Owned refinement scratch. Allocated by prepare! so the refinement step itself
# writes into existing destinations.


# Internal: per-cache refine storage helper. Kept simple and allocation-free on
# the hot refinement path after the first prepare.
function _cache_refine_storage(cache::AbstractFactorCache, size_of_b)
    storage = cache.refine
    if storage === nothing || size(storage.residual) != size_of_b ||
            storage.precision_bits != cache.precision_bits
        storage = CacheRefineStorage(
            owned_zeros(BigFloat, size_of_b...; precision_bits = cache.precision_bits),
            owned_zeros(BigFloat, size_of_b...; precision_bits = cache.precision_bits),
            cache.precision_bits,
        )
        cache.refine = storage
    end
    return storage
end

function _refine_once_cached(cache, A, x, b, residual, p)
    storage = cache.refine
    residual = storage.residual
    correction = storage.correction
    backend = cache.backend
    # residual = b - A*x into owned residual
    residual!(backend, NoTrans, A, x, b, residual)
    before = normwise_backward_error(backend, NoTrans, A, x, b, residual)
    # correction = residual (factor precision), solved by the factor
    _cache_rhs_into!(correction, residual, "refine_once!")
    _cache_solve_inplace!(cache, correction)
    # x += correction
    _cache_axpy!(x, correction, cache)
    residual!(backend, NoTrans, A, x, b, residual)
    after = normwise_backward_error(backend, NoTrans, A, x, b, residual)
    return (
        x = x,
        backend = backend,
        factor_precision = p,
        backward_error_before = before,
        backward_error_after = after,
    )
end

function _cache_axpy!(x::AbstractArray{BigFloat}, correction::AbstractArray{BigFloat}, cache)
    p = cache.precision_bits
    one_value = cache.scalars === nothing ? BigFloat(1; precision = p) : cache.scalars.one
    X = reshape(x, length(x))
    D = reshape(correction, length(correction))
    return _axpy!(cache.backend, one_value, D, X, p)
end

# In-place factor solve: `x` is both the right-hand side (on entry) and the
# solution (on exit). Used by refinement where the correction buffer is the
# destination; it does not copy, so the buffer must already carry the RHS.
function _cache_solve_inplace!(cache::BFLACholeskyCache, x::AbstractVecOrMat{BigFloat})
    workspace, worker = _cache_solve_workspace(cache, 1)
    _cholesky_solve!(
        cache.backend, cache.factors, cache.triangle, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

function _cache_solve_inplace!(cache::BFLALUCache, x::AbstractVecOrMat{BigFloat})
    workspace, worker = _cache_solve_workspace(cache, 1)
    _lu_solve!(
        cache.backend, cache.factors, cache.pivots, cache.precision_bits,
        x, workspace, worker,
    )
    return x
end

function _cache_solve_inplace!(cache::BFLALDLTCache, x::AbstractVecOrMat{BigFloat})
    workspace, worker = _cache_solve_workspace(cache, 1)
    _ldlt_solve!(cache.backend, cache, x, workspace, worker)
    return x
end

function _cache_solve_inplace!(cache::BFLARRQRCache, x::AbstractVecOrMat{BigFloat})
    workspace, worker = _cache_solve_workspace(cache, 1)
    _qr_solve!(cache.backend, cache, x, workspace, worker)
    return x
end
