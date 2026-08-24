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
re-factorizes into the *same* owned storage (no factor deep-copy).

**RHS-shape lifecycle:** within one cache lifetime the RHS container dimension
and column count are fixed (LinearSolve sizes `cache.u` once). RHS *values* may
be updated, and `A` may be refreshed, but changing the RHS shape or column count
(e.g. vector → matrix) or the precision requires a fresh `LinearSolve.init` /
`reinit!` that re-shapes `cache.u`.
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

The adapter is built directly on a reusable BFLA factor cache — there is no
second, allocating factor copy. `cache` owns the factor matrix and workspace;
`factorize!` and `solve_trusted!` write into existing owned destinations, so an
RHS-only solve allocates no new BigFloat objects and a matrix refresh reuses the
same factor storage. The solution-ownership fields let the adapter re-verify
`cache.u` (shape, precision, array identity) so a reinit/re-shape never silently
corrupts a shared `fill!` destination.
"""
mutable struct _BFLALinearCache{C<:BFLA.AbstractFactorCache}
    cache::Union{Nothing,C}
    u_ready::Bool
    u_array_id::UInt
    u_shape::Tuple
    u_precision::Int
end

_BFLALinearCache{C}() where {C} = _BFLALinearCache{C}(nothing, false, 0, (), 0)

function LinearSolve.init_cacheval(
        alg::BigFloatLU, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    C = BFLA.BFLALUCache{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{C}()
end

function LinearSolve.init_cacheval(
        alg::BigFloatCholesky, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    C = BFLA.BFLACholeskyCache{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{C}()
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

# Prepare the reusable cache for this problem if not already at the right
# size/precision; then factorize the current `A` into its owned storage. The
# factor matrix object identity is preserved across matrix refreshes (in-place
# re-factorization); nothing is deep-copied.
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
    return c
end

# Re-verify the LinearSolve solution `cache.u` against the adapter's recorded
# shape, precision, and array identity. If the array was replaced, reshaped, or
# re-precisioned (e.g. `reinit!`, vector->matrix, 128->256 bit), re-own every
# slot so the trusted `solve_trusted!` path never writes into shared `fill!`
# storage. On the steady repeated-solve path nothing changes and nothing
# allocates.
function _ensure_solution_owned!(state::_BFLALinearCache, u, p::Int)
    p_u = try
        BFLA._require_precision(BFLA._check_precision(u), "LinearSolve u")
    catch
        -1
    end
    if !state.u_ready || objectid(u) != state.u_array_id ||
            size(u) != state.u_shape || p_u != state.u_precision ||
            state.u_precision != p
        @inbounds for index in eachindex(u)
            u[index] = BigFloat(0; precision = p)
        end
        state.u_ready = true
        state.u_array_id = objectid(u)
        state.u_shape = size(u)
        state.u_precision = p
    end
    return nothing
end

@inline function _solve_common!(cache, alg, state)
    c = state.cache
    _ensure_solution_owned!(state, cache.u, BFLA.factor_precision(c))
    # Trusted solve: `u` is owned at the factor precision, so `solve_trusted!`
    # copies `cache.b` into the existing `u` objects and solves in place with no
    # new BigFloat allocation.
    BFLA.solve_trusted!(cache.u, c, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Success,
    )
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::BigFloatLU; kwargs...
    )
    state = cache.cacheval
    if cache.isfresh
        state.cache = _ensure_factorized!(state, alg, cache.A, cache.b)
        if !BFLA.issuccess(state.cache)
            return _failure(alg, cache)
        end
        cache.isfresh = false
    end
    c = state.cache
    c === nothing && return _failure(alg, cache)
    BFLA.issuccess(c) || return _failure(alg, cache)
    return _solve_common!(cache, alg, state)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::BigFloatCholesky; kwargs...
    )
    state = cache.cacheval
    if cache.isfresh
        state.cache = _ensure_factorized!(state, alg, cache.A, cache.b)
        if !BFLA.issuccess(state.cache)
            return _failure(alg, cache)
        end
        cache.isfresh = false
    end
    c = state.cache
    c === nothing && return _failure(alg, cache)
    BFLA.issuccess(c) || return _failure(alg, cache)
    return _solve_common!(cache, alg, state)
end

end
