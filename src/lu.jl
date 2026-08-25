# Dense square LU with partial (row-maximum) pivoting.

"""
    BFLALUFactor{M,B} <: AbstractBFLAFactor

Packed partial-pivoting factorization `P*A = L*U` of a square dense matrix.
The strict lower triangle stores the unit-lower `L` multipliers and the upper
triangle stores `U`.

`pivots[k]` is the row swapped with row `k` at elimination step `k`.
`perm[i]` is the original row occupying factor row `i`, so `A[perm, :] = L*U`.
The factor records its producing backend and storage precision; every solve
dispatches through that backend.
"""
struct BFLALUFactor{
    M<:AbstractMatrix{BigFloat},
    B<:AbstractBFLABackend,
} <: AbstractBFLAFactor
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    pivots::Vector{Int}
    perm::Vector{Int}
end

factor_matrix(F::BFLALUFactor) = F.factors
factor_backend(F::BFLALUFactor) = F.backend
factor_precision(F::BFLALUFactor) = F.precision_bits
factor_status(F::BFLALUFactor) = F.status
factor_kind(::BFLALUFactor) = :lu
factor_triangle(::BFLALUFactor) = nothing
factor_perm(F::BFLALUFactor) = copy(F.perm)
issuccess(F::BFLALUFactor) = F.status.kind === :success

"""
    factor_pivots(F::BFLALUFactor) -> Vector{Int}

Partial-pivot row swaps. At elimination step `k`, rows `k` and
`factor_pivots(F)[k]` were exchanged.
"""
factor_pivots(F::BFLALUFactor) = copy(F.pivots)

"""
    factor_diagnostics(F::BFLALUFactor) -> NamedTuple

Report row-swap count, final permutation, and optional failure position.
These are numerical facts only; no fallback policy is implied.
"""
function factor_diagnostics(F::BFLALUFactor)
    return (
        factor_kind = factor_kind(F),
        row_swap_count = count(k -> F.pivots[k] != k, eachindex(F.pivots)),
        permutation = copy(F.perm),
        failure_position = factor_failure_position(F),
    )
end

Base.size(F::BFLALUFactor) = size(F.factors)
Base.size(F::BFLALUFactor, dimension::Integer) = size(F.factors, dimension)
Base.eltype(::BFLALUFactor{M,B}) where {M,B} = BigFloat

"""
    lu!(backend, A; check=true) -> BFLALUFactor

Factor square `A` in place using partial row pivoting and borrow its storage.
Non-finite input fails closed. A zero pivot throws `SingularException` when
`check=true`; `check=false` returns a factor with `FactorStatus(:singular, k)`.
Failure may leave `A` partially overwritten; no rollback is promised.
"""
function lu! end

function lu!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    check::Bool=true,
)
    _require_square(A, "lu!")
    p = _require_precision(_check_precision(A), "lu!")
    n = size(A, 1)
    if !_all_finite(A)
        check && throw(DomainError(A, "lu!: input contains non-finite entries"))
        identity = collect(1:n)
        return BFLALUFactor(
            A,
            backend,
            p,
            FactorStatus(:nonfinite, nothing),
            copy(identity),
            identity,
        )
    end
    info, pivots = _lu!(backend, A, p)
    perm = _lu_permutation(pivots)
    if !_all_finite(A)
        check && throw(DomainError(
            A, "lu!: factorization produced non-finite entries",
        ))
        return BFLALUFactor(
            A,
            backend,
            p,
            FactorStatus(:nonfinite, nothing),
            pivots,
            perm,
        )
    end
    if info != 0
        check && throw(LinearAlgebra.SingularException(info))
        return BFLALUFactor(
            A, backend, p, FactorStatus(:singular, info), pivots, perm,
        )
    end
    return BFLALUFactor(A, backend, p, SUCCESS_STATUS, pivots, perm)
end

"""
    try_lu!(backend, A) -> Union{BFLALUFactor,Nothing}

Like `lu!(backend, A; check=false)`, returning `nothing` on singular or
non-finite input. The input is still an in-place target and may be modified.
"""
function try_lu! end

function try_lu!(backend::AbstractBFLABackend, A::AbstractMatrix{BigFloat})
    F = lu!(backend, A; check=false)
    return issuccess(F) ? F : nothing
end

"""
    lu(backend, A; check=true) -> BFLALUFactor

Allocating partial-pivoting LU. `A` is ownership-safe deep-copied before
factorization, so the returned factor owns storage and `A` remains unchanged.
"""
function lu end

function lu(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    check::Bool=true,
)
    _require_square(A, "lu")
    p = _require_precision(_check_precision(A), "lu")
    return lu!(backend, owned_copy(A; precision_bits=p); check=check)
end

function _lu_permutation(pivots::Vector{Int})
    perm = collect(eachindex(pivots))
    for k in eachindex(pivots)
        pivot = pivots[k]
        perm[k], perm[pivot] = perm[pivot], perm[k]
    end
    return perm
end

function _lu_ldiv!(
    F::BFLALUFactor,
    rhs::AbstractVecOrMat{BigFloat},
    trusted::Bool,
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
    operation::AbstractString,
)
    issuccess(F) || throw(LinearAlgebra.SingularException(
        F.status.position === nothing ? 0 : F.status.position,
    ))
    n = size(F.factors, 1)
    size(rhs, 1) == n || throw(DimensionMismatch(
        "$operation: right-hand side dimensions differ",
    ))
    _require_no_alias(rhs, F.factors, operation)
    if trusted
        _validate_trusted_rhs_precision(F, operation, rhs)
    else
        _validate_factor_precision(F, operation, rhs)
        _validate_factor_metadata(F, operation)
        _all_finite(F.factors) || throw(DomainError(
            F, "$operation: factor storage contains non-finite entries",
        ))
    end
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: right-hand side contains non-finite entries",
    ))
    _validate_solve_workspace(
        workspace, workspace_worker, F.precision_bits, operation,
    )
    _lu_solve!(
        F.backend,
        F.factors,
        F.pivots,
        F.precision_bits,
        rhs,
        workspace,
        workspace_worker,
    )
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: solve produced non-finite entries",
    ))
    return rhs
end

function ldiv!(
    F::BFLALUFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _lu_ldiv!(F, rhs, false, workspace, workspace_worker, "ldiv!")
end

function ldiv_trusted!(
    F::BFLALUFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _lu_ldiv!(
        F, rhs, true, workspace, workspace_worker, "ldiv_trusted!",
    )
end

function solve!(
    F::BFLALUFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return ldiv!(
        F, rhs; workspace=workspace, workspace_worker=workspace_worker,
    )
end

function solve(
    F::BFLALUFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return ldiv!(
        F,
        owned_copy(rhs);
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
end

function _lu_abs_to!(destination::BigFloat, source::BigFloat)
    MA.operate_to!(destination, copy, source)
    signbit(destination) && MA.operate!(-, destination)
    return destination
end

function _lu!(::NativeBackend, A::AbstractMatrix{BigFloat}, p::Int)
    n = size(A, 1)
    pivots = collect(1:n)
    maximum_value = BigFloat(0; precision = p)
    candidate = BigFloat(0; precision = p)
    product = BigFloat(0; precision = p)
    @inbounds for k in 1:n
        pivot = k
        _lu_abs_to!(maximum_value, A[k, k])
        for i in (k + 1):n
            _lu_abs_to!(candidate, A[i, k])
            if candidate > maximum_value
                MA.operate_to!(maximum_value, copy, candidate)
                pivot = i
            end
        end
        pivots[k] = pivot
        iszero(maximum_value) && return k, pivots
        if pivot != k
            for j in 1:n
                A[k, j], A[pivot, j] = A[pivot, j], A[k, j]
            end
        end
        for i in (k + 1):n
            _mpfr_div!(A[i, k], A[i, k], A[k, k])
            for j in (k + 1):n
                MA.operate_to!(product, *, A[i, k], A[k, j])
                MA.operate!(-, A[i, j], product)
            end
        end
    end
    return 0, pivots
end

function _lu!(::GenericBackend, A::AbstractMatrix{BigFloat}, p::Int)
    return _with_precision(p) do
        F = LinearAlgebra.lu!(A, LinearAlgebra.RowMaximum(); check=false)
        return F.info, collect(F.ipiv)
    end
end

function _lu_solve!(
    ::NativeBackend,
    A::AbstractMatrix{BigFloat},
    pivots::Vector{Int},
    p::Int,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
)
    n = size(A, 1)
    nrhs = length(rhs) ÷ n
    product = _solve_scratch(workspace, workspace_worker, 1, p)
    @inbounds for column in 1:nrhs
        base = (column - 1) * n
        for k in 1:n
            pivot = pivots[k]
            rhs[base + k], rhs[base + pivot] = rhs[base + pivot], rhs[base + k]
        end
        for i in 1:n
            for k in 1:(i - 1)
                MA.operate_to!(product, *, A[i, k], rhs[base + k])
                MA.operate!(-, rhs[base + i], product)
            end
        end
        for i in n:-1:1
            for k in (i + 1):n
                MA.operate_to!(product, *, A[i, k], rhs[base + k])
                MA.operate!(-, rhs[base + i], product)
            end
            _mpfr_div!(rhs[base + i], rhs[base + i], A[i, i])
        end
    end
    return rhs
end

function _lu_solve!(
    ::GenericBackend,
    A::AbstractMatrix{BigFloat},
    pivots::Vector{Int},
    p::Int,
    rhs::AbstractVecOrMat{BigFloat},
    ::Union{Nothing,BFLAWorkspace},
    ::Int,
)
    return _with_precision(p) do
        F = LinearAlgebra.LU(A, pivots, 0)
        LinearAlgebra.ldiv!(F, rhs)
    end
end
