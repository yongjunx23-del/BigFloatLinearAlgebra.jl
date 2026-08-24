module BigFloatLinearSolveExt

import BigFloatLinearAlgebra as BFLA
import LinearSolve
import MutableArithmetics as MA
import SciMLBase

"""
    BigFloatLU(; backend = BFLA.DEFAULT_BACKEND)

`LinearSolve.jl` algorithm that uses BFLA's ownership-safe BigFloat LU
factorization. The matrix is copied by BFLA before factorization, and the
factor is reused until LinearSolve marks the cache as fresh again. The
LinearSolve cache aliases caller-owned `A` and `b`, but BFLA never mutates
either input. After mutating `A` in place, assign it back to `cache.A` or call
`reinit!(cache; A = A)` so the factorization is refreshed.
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
factorization. Only the selected authoritative triangle is read by BFLA. The
same input-aliasing and explicit matrix-refresh contract as [`BigFloatLU`](@ref)
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

# Direct subtypes must provide all four traits in the package that defines the
# algorithm. Aliasing avoids LinearSolve's ordinary shallow `copy` of mutable
# BigFloat elements; BFLA's allocating factors still own deep matrix copies.
LinearSolve.needs_concrete_A(::Union{BigFloatLU,BigFloatCholesky}) = true
LinearSolve.needs_square_A(::Union{BigFloatLU,BigFloatCholesky}) = true
LinearSolve.default_alias_A(
    ::Union{BigFloatLU,BigFloatCholesky}, ::Any, ::Any,
) = true
LinearSolve.default_alias_b(
    ::Union{BigFloatLU,BigFloatCholesky}, ::Any, ::Any,
) = true

"""Typed cache slot kept by a LinearSolve `LinearCache`."""
mutable struct _BFLALinearCache{F<:BFLA.AbstractBFLAFactor}
    factor::Union{Nothing,F}
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

function LinearSolve.init_cacheval(
        alg::BigFloatLU, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    # Do not factor at init time: `LinearSolve.init` marks the cache fresh and
    # the first solve owns the one factorization that establishes cacheval.
    factor_type = BFLA.BFLALUFactor{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{factor_type}(nothing)
end

function LinearSolve.init_cacheval(
        alg::BigFloatCholesky, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assumptions,
    )
    factor_type = BFLA.BFLACholeskyFactor{Matrix{BigFloat},typeof(alg.backend)}
    return _BFLALinearCache{factor_type}(nothing)
end

@inline function _failure(alg, cache)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Failure,
    )
end

# LinearSolve initializes `cache.u` with `similar` followed by `fill!`, so its
# BigFloat elements can carry the ambient precision rather than the problem's
# explicit precision. Replace every slot after validating the source, instead
# of requiring the destination's stale precision to match.
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
        destination[index] = MA.mutable_copy(source[index])
    end
    return destination
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::BigFloatLU; kwargs...
    )
    state = cache.cacheval
    if cache.isfresh
        state.factor = nothing
        factor = BFLA.lu(alg.backend, cache.A; check = false)
        if !BFLA.issuccess(factor)
            return _failure(alg, cache)
        end
        state.factor = factor
        cache.isfresh = false
    end

    factor = state.factor
    factor === nothing && return _failure(alg, cache)
    _copy_rhs_owned!(cache.u, cache.b, BFLA.factor_precision(factor))
    BFLA.ldiv_trusted!(factor, cache.u)
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
        state.factor = nothing
        factor = BFLA.cholesky(
            alg.backend, cache.A;
            triangle = alg.triangle,
            check = false,
        )
        if !BFLA.issuccess(factor)
            return _failure(alg, cache)
        end
        state.factor = factor
        cache.isfresh = false
    end

    factor = state.factor
    factor === nothing && return _failure(alg, cache)
    _copy_rhs_owned!(cache.u, cache.b, BFLA.factor_precision(factor))
    BFLA.ldiv_trusted!(factor, cache.u)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = SciMLBase.ReturnCode.Success,
    )
end

end
