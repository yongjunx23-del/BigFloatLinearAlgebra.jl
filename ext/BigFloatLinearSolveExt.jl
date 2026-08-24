module BigFloatLinearSolveExt

import BigFloatLinearAlgebra as BFLA
import LinearSolve
import MutableArithmetics as MA
import SciMLBase

"""
    BigFloatLU(; backend = BFLA.DEFAULT_BACKEND)

`LinearSolve.jl` algorithm that uses BFLA's ownership-safe BigFloat LU
factorization through a reusable, precision-specific factor cache. The factor
matrix is owned by the cache and reused across right-hand-side-only updates
until LinearSolve marks the cache fresh again. After mutating `A` in place,
assign it back to `cache.A` or call `reinit!(cache; A = A)` so the cache
re-factorizes into the same owned storage.
"""
struct BigFloatLU{B<:BFLA.AbstractBFLABackend} <:
    LinearSolve.SciMLLinearSolveAlgorithm
    backend::B
end

BigFloatLU(; backend::BFLA.AbstractBFLABackend = BFLA.DEFAULT_BACKEND) =
    BigFloatLU{typeof(backend)}(backend)

"""
    BigFloatCholesky(; backend = BFLA.DEFAULT_BACKEND, triangle = BFLA.Lower)

`LinearSolve.jl` algorithm that uses BFLA's ownership-safe BigFloat Cholesky
factorization. Only the selected authoritative triangle is read. The
input-aliasing and explicit matrix-refresh contract of [`BigFloatLU`](@ref)
applies.
"""
struct BigFloatCholesky{B<:BFLA.AbstractBFLABackend} <:
    LinearSolve.SciMLLinearSolveAlgorithm
    backend::B
    triangle::BFLA.Triangle
end

BigFloatCholesky(; backend::BFLA.AbstractBFLABackend = BFLA.DEFAULT_BACKEND,
                 triangle::BFLA.Triangle = BFLA.Lower) =
    BigFloatCholesky{typeof(backend)}(backend, triangle)
BigFloatCholesky(backend::BFLA.AbstractBFLABackend) =
    BigFloatCholesky{typeof(backend)}(backend, BFLA.Lower)

LinearSolve.needs_concrete_A(::Union{BigFloatLU,BigFloatCholesky}) = true
LinearSolve.needs_square_A(::Union{BigFloatLU,BigFloatCholesky}) = true
LinearSolve.default_alias_A(
    ::Union{BigFloatLU,BigFloatCholesky}, ::Any, ::Any,
) = true
LinearSolve.default_alias_b(
    ::Union{BigFloatLU,BigFloatCholesky}, ::Any, ::Any,
) = true

"""Typed cache slot kept by a LinearSolve `LinearCache`.

Holds a BFLA reusable factor cache (the hot path) plus a compatibility factor
handle exposing the ordinary `AbstractBFLAFactor` API. The `cache` owns its
factor matrix and workspace; `factorize!` and `solve!` write into existing owned
destinations, so repeated RHS-only solves allocate no new BigFloat objects. The
`factor` handle is rebuilt only when the matrix is refreshed, never on an
RHS-only update.
"""
mutable struct _BFLALinearCache{F<:BFLA.AbstractBFLAFactor}
    cache::Union{Nothing,BFLA.AbstractFactorCache}
    factor::Union{Nothing,F}
    prepared::Bool
    u_repaired::Bool
end

_BFLALinearCache{F}() where {F} = _BFLALinearCache{F}(nothing, nothing, false, false)

# Build a compatibility factor handle that *owns fresh copies* of the cache's
# factor data. Each fresh/refreshed factorization therefore yields a distinct
# factor object with the ordinary allocating-factor semantics. This handle is
# rebuilt only when the matrix is (re)factorized; the repeated RHS-only solve
# path never touches it, so it adds no allocation to the hot path.
function _cache_handle(c::BFLA.BFLALUCache)
    return BFLA.BFLALUFactor(
        BFLA.owned_copy(c.factors; precision_bits = c.precision_bits),
        c.backend, c.precision_bits, c.status, copy(c.pivots), copy(c.perm),
    )
end
function _cache_handle(c::BFLA.BFLACholeskyCache)
    return BFLA.BFLACholeskyFactor(
        BFLA.owned_copy(c.factors; precision_bits = c.precision_bits),
        c.backend, c.triangle, c.precision_bits, c.status,
    )
end

function LinearSolve.init_cacheval(
        alg::BigFloatLU, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    F = BFLA.BFLALUFactor{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{F}()
end

function LinearSolve.init_cacheval(
        alg::BigFloatCholesky, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    F = BFLA.BFLACholeskyFactor{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{F}()
end

@inline function _failure(alg, cache)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Failure,
    )
end

# LinearSolve 3.x creates a vector `u0` for a matrix RHS. Supplying the owned
# matrix here gives all supported LinearSolve versions the same batched-RHS
# cache shape and explicit MPFR precision.
function SciMLBase.init(
        prob::SciMLBase.LinearProblem{Nothing,IIP,AT,BT},
        alg::Union{BigFloatLU,BigFloatCholesky},
        args...;
        kwargs...,
    ) where {
        IIP,
        AT<:AbstractMatrix{BigFloat},
        BT<:AbstractMatrix{BigFloat},
    }
    precision_bits = BFLA._require_precision(
        BFLA._check_precision(prob.A, prob.b), "LinearSolve init",
    )
    u0 = BFLA.owned_zeros(
        BigFloat, size(prob.A, 2), size(prob.b, 2);
        precision_bits = precision_bits,
    )
    return SciMLBase.init(
        SciMLBase.remake(prob; u0 = u0), alg, args...; kwargs...,
    )
end

function _new_cache(alg::BigFloatLU)
    return BFLA.BFLALUCache(alg.backend)
end
function _new_cache(alg::BigFloatCholesky)
    return BFLA.BFLACholeskyCache(alg.backend; triangle = alg.triangle)
end

# Copy an independently-owned RHS into the existing owned solution elements,
# preserving destination object identity (no element replacement, no allocation
# of new BigFloat objects on the hot path).
function _copy_rhs_owned!(
        destination::AbstractArray{BigFloat},
        source::AbstractArray{BigFloat},
        precision_bits::Int,
    )
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "LinearSolve solve!: solution and right-hand side axes differ",
    ))
    Base.mightalias(destination, source) && throw(ArgumentError(
        "LinearSolve solve!: solution must not alias the right-hand side",
    ))
    source_precision = BFLA._require_precision(
        BFLA._check_precision(source), "LinearSolve solve!",
    )
    source_precision == precision_bits || throw(BFLA.PrecisionMismatch(
        precision_bits, source_precision, nothing,
    ))
    @inbounds for index in eachindex(destination, source)
        MA.operate_to!(destination[index], copy, source[index])
    end
    return destination
end

# Prepare the reusable cache for this problem if not already at the right
# size/precision; then factorize the current `A` into its owned storage and
# rebuild the compatibility factor handle.
function _ensure_factorized!(cache::_BFLALinearCache, alg, A, b)
    c = cache.cache
    p = BFLA._require_precision(
        BFLA._check_precision(A, b), "LinearSolve factorize",
    )
    n = size(A, 1)
    if c === nothing || c.precision_bits != p || c.n != n
        c = _new_cache(alg)
        BFLA.prepare!(c, n, p; nrhs = size(b, 2))
        cache.cache = c
    end
    BFLA.factorize!(c, A)
    cache.factor = _cache_handle(c)
    return c
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::BigFloatLU; kwargs...
    )
    state = cache.cacheval
    if cache.isfresh
        state.cache = _ensure_factorized!(state, alg, cache.A, cache.b)
        if !BFLA.issuccess(state.cache)
            state.factor = nothing
            return _failure(alg, cache)
        end
        cache.isfresh = false
    end
    c = state.cache
    c === nothing && return _failure(alg, cache)
    BFLA.issuccess(c) || return _failure(alg, cache)
    _repair_solution_owned!(state, cache, BFLA.factor_precision(c))
    # Solve in place: `solve!(u, cache, b)` copies b into the existing owned
    # `u` elements and overwrites them with the solution, so repeated RHS-only
    # solves reuse the factor and allocate no new BigFloat objects.
    BFLA.solve!(cache.u, c, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Success,
    )
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::BigFloatCholesky; kwargs...
    )
    state = cache.cacheval
    if cache.isfresh
        state.cache = _ensure_factorized!(state, alg, cache.A, cache.b)
        if !BFLA.issuccess(state.cache)
            state.factor = nothing
            return _failure(alg, cache)
        end
        cache.isfresh = false
    end
    c = state.cache
    c === nothing && return _failure(alg, cache)
    BFLA.issuccess(c) || return _failure(alg, cache)
    _repair_solution_owned!(state, cache, BFLA.factor_precision(c))
    BFLA.solve!(cache.u, c, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Success,
    )
end
# LinearSolve initializes the solution with `similar`+`fill!`, which installs a
# shared (non-owned) object in every slot and can carry the ambient precision.
# Repair it once so `_cache_rhs_into!` can take its zero-allocation write-into-
# existing path on every later solve. This one-time repair is the documented
# warm-up cost; it is not part of the repeated numeric hot path.
function _repair_solution_owned!(state::_BFLALinearCache, cache, p::Int)
    state.u_repaired && return cache.u
    @inbounds for index in eachindex(cache.u)
        cache.u[index] = BigFloat(0; precision = p)
    end
    state.u_repaired = true
    return cache.u
end

end
