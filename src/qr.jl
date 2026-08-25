# Rank-revealing column-pivoted Householder QR.

"""
    BFLAQRFactor{M,B} <: AbstractBFLAFactor

Column-pivoted QR factorization `A * P = Q * R`, where `Q` is orthogonal, `R`
is upper triangular (upper trapezoidal for non-square `A`), and `P` is the
column permutation `jpvt`.

`factors` stores `R` in the upper triangle and the Householder vectors in the
strictly lower triangle; `tau` holds the Householder scalars.
"""
struct BFLAQRFactor{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractBFLAFactor
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    tau::Vector{BigFloat}
    jpvt::Vector{Int}
    rank::Int
    # `tolerance` is the 0.1.0 spelling and remains an absolute tolerance
    # alias for callers that still inspect the old field.  The explicit
    # fields below are the authoritative RRQR rank policy metadata.
    tolerance::BigFloat
    atol::BigFloat
    rtol::BigFloat
    reference_scale::BigFloat
    effective_threshold::BigFloat
end

_factor_precision_operands(F::BFLAQRFactor) = (
    F.tau,
    F.tolerance,
    F.atol,
    F.rtol,
    F.reference_scale,
    F.effective_threshold,
)

# Keep the 0.1.0 positional constructor usable by existing callers and by
# factors serialized by that release.  Such a factor has absolute-only rank
# metadata, which is exactly the old `tol` policy.
function BFLAQRFactor(
    factors::M,
    backend::B,
    precision_bits::Int,
    status::FactorStatus,
    tau::Vector{BigFloat},
    jpvt::Vector{Int},
    rank::Int,
    tolerance::BigFloat,
) where {M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend}
    atol = MA.mutable_copy(tolerance)
    rtol = BigFloat(0; precision = precision_bits)
    reference_scale = BigFloat(0; precision = precision_bits)
    effective_threshold = MA.mutable_copy(tolerance)
    return BFLAQRFactor(
        factors,
        backend,
        precision_bits,
        status,
        tau,
        jpvt,
        rank,
        MA.mutable_copy(tolerance),
        atol,
        rtol,
        reference_scale,
        effective_threshold,
    )
end

factor_matrix(F::BFLAQRFactor) = F.factors
factor_backend(F::BFLAQRFactor) = F.backend
factor_precision(F::BFLAQRFactor) = F.precision_bits
factor_status(F::BFLAQRFactor) = F.status
factor_kind(::BFLAQRFactor) = :rrqr
factor_triangle(::BFLAQRFactor) = nothing
issuccess(F::BFLAQRFactor) = F.status.kind === :success

"""
    factor_rank(F) -> Int

Numerical rank of the factorization.
"""
function factor_rank(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_rank")
    return F.rank
end

"""
    factor_jpvt(F) -> Vector{Int}

Column permutation (`A * P = Q * R`).
"""
function factor_jpvt(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_jpvt")
    return copy(F.jpvt)
end

"""
    factor_Rdiag(F) -> Vector{BigFloat}

Diagonal entries of the `R` factor, in pivot order.
"""
function factor_Rdiag(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_Rdiag")
    return [
        MA.mutable_copy(F.factors[i, i])
        for i in 1:min(size(F.factors, 1), size(F.factors, 2))
    ]
end

"""
    factor_tolerance(F) -> BigFloat

The absolute RRQR rank tolerance (the 0.1.0 spelling; alias of
[`factor_rank_atol`](@ref)).
"""
function factor_tolerance(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_tolerance")
    return MA.mutable_copy(F.tolerance)
end

"""
    factor_rank_atol(F) -> BigFloat

Return the absolute component of the RRQR numerical-rank policy.  The result
is a defensive copy.
"""
function factor_rank_atol(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_rank_atol")
    return MA.mutable_copy(F.atol)
end

"""
    factor_rank_rtol(F) -> BigFloat

Return the relative component of the RRQR numerical-rank policy.  The result
is a defensive copy.
"""
function factor_rank_rtol(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_rank_rtol")
    return MA.mutable_copy(F.rtol)
end

"""
    factor_rank_scale(F) -> BigFloat

Return the scale used as the reference for the relative rank threshold.  For
RRQR this is the largest input-column 2-norm, computed before the packed
factor storage is overwritten.  The result is a defensive copy.
"""
function factor_rank_scale(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_rank_scale")
    return MA.mutable_copy(F.reference_scale)
end

"""
    factor_rank_threshold(F) -> BigFloat

    Return `max(atol, rtol * reference_scale)`, the threshold used by the stored
RRQR rank decision.  The result is a defensive copy.
"""
function factor_rank_threshold(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_rank_threshold")
    return MA.mutable_copy(F.effective_threshold)
end

"""
    factor_diagnostics(F::BFLAQRFactor) -> NamedTuple

Return RRQR rank-policy metadata.  Mutable numeric values and the pivot vector
are copied so inspecting diagnostics cannot mutate the factor.
"""
function factor_diagnostics(F::BFLAQRFactor)
    _validate_factor_integrity!(F, "factor_diagnostics")
    diagonal = factor_Rdiag(F)
    minimum_accepted = nothing
    absolute = BigFloat(0; precision = F.precision_bits)
    for i in 1:F.rank
        _factor_abs_to!(absolute, diagonal[i])
        if minimum_accepted === nothing || absolute < minimum_accepted
            minimum_accepted = MA.mutable_copy(absolute)
        end
    end
    next_rejected = if F.rank < length(diagonal)
        _factor_abs_to!(absolute, diagonal[F.rank + 1])
        MA.mutable_copy(absolute)
    else
        nothing
    end
    return (
        factor_kind = factor_kind(F),
        rank = factor_rank(F),
        atol = factor_rank_atol(F),
        rtol = factor_rank_rtol(F),
        reference_scale = factor_rank_scale(F),
        effective_threshold = factor_rank_threshold(F),
        R_diagonal = diagonal,
        min_accepted_abs_Rdiag = minimum_accepted,
        next_rejected_abs_Rdiag = next_rejected,
        permutation = factor_jpvt(F),
        failure_position = factor_failure_position(F),
    )
end

function _qr_validate_tolerance(
    value::BigFloat,
    name::AbstractString,
    operation::AbstractString,
)
    isfinite(value) || throw(DomainError(
        value, "$operation: $name must be finite",
    ))
    value >= 0 || throw(DomainError(
        value, "$operation: $name must be nonnegative",
    ))
    return nothing
end

function _qr_default_rtol(p::Int, m::Int, n::Int)
    # The dimension factor is the usual backward-error guard. Construct every
    # quantity directly at factor precision, independent of ambient precision.
    epsilon = BigFloat(0; precision = p)
    _mpfr_set_ui_2exp!(epsilon, 1, 1 - p)
    result = BigFloat(0; precision = p)
    MA.operate_to!(result, *, BigFloat(max(m, n); precision = p), epsilon)
    return result
end

function _qr_segment_norm!(
    result::BigFloat,
    A::AbstractMatrix{BigFloat},
    first_row::Int,
    column::Int,
    scale::BigFloat,
    sumsq::BigFloat,
    absolute::BigFloat,
    ratio::BigFloat,
    term::BigFloat,
    one_value::BigFloat,
)
    MA.operate!(zero, scale)
    MA.operate_to!(sumsq, copy, one_value)
    @inbounds for row in first_row:size(A, 1)
        _factor_abs_to!(absolute, A[row, column])
        iszero(absolute) && continue
        if scale < absolute
            _mpfr_div!(ratio, scale, absolute)
            MA.operate_to!(term, *, ratio, ratio)
            MA.operate!(*, term, sumsq)
            MA.operate!(+, term, one_value)
            MA.operate_to!(sumsq, copy, term)
            MA.operate_to!(scale, copy, absolute)
        else
            _mpfr_div!(ratio, absolute, scale)
            MA.buffered_operate!(term, MA.add_mul, sumsq, ratio, ratio)
        end
    end
    if iszero(scale)
        MA.operate!(zero, result)
    else
        _mpfr_sqrt!(ratio, sumsq)
        MA.operate_to!(result, *, scale, ratio)
    end
    return result
end

function _qr_reference_scale(A::AbstractMatrix{BigFloat}, p::Int)
    scale = BigFloat(0; precision = p)
    colnorm = BigFloat(0; precision = p)
    norm_scale = BigFloat(0; precision = p)
    sumsq = BigFloat(0; precision = p)
    absolute = BigFloat(0; precision = p)
    ratio = BigFloat(0; precision = p)
    term = BigFloat(0; precision = p)
    one_value = BigFloat(1; precision = p)
    @inbounds for j in axes(A, 2)
        _qr_segment_norm!(
            colnorm, A, first(axes(A, 1)), j, norm_scale, sumsq,
            absolute, ratio, term, one_value,
        )
        colnorm > scale && MA.operate_to!(scale, copy, colnorm)
    end
    return scale
end

function _qr_rank_from_factors(A::AbstractMatrix{BigFloat}, threshold::BigFloat)
    rank = 0
    absolute = BigFloat(0; precision = precision(threshold))
    @inbounds for i in 1:min(size(A, 1), size(A, 2))
        MA.operate_to!(absolute, copy, A[i, i])
        signbit(absolute) && MA.operate!(-, absolute)
        absolute > threshold || break
        rank += 1
    end
    return rank
end

function _qr_rank_policy(
    A::AbstractMatrix{BigFloat},
    p::Int,
    tol::Union{Nothing,BigFloat},
    atol::Union{Nothing,BigFloat},
    rtol::Union{Nothing,BigFloat},
    operation::AbstractString,
)
    tol === nothing || (atol === nothing && rtol === nothing) ||
        throw(ArgumentError(
            "$operation: legacy tol cannot be combined with atol or rtol",
        ))
    if tol === nothing
        absolute = atol === nothing ? BigFloat(0; precision = p) : MA.mutable_copy(atol)
        relative = rtol === nothing ?
            _qr_default_rtol(p, size(A, 1), size(A, 2)) : MA.mutable_copy(rtol)
    else
        absolute = MA.mutable_copy(tol)
        relative = BigFloat(0; precision = p)
    end
    _qr_validate_tolerance(absolute, "atol", operation)
    _qr_validate_tolerance(relative, "rtol", operation)
    scale = _qr_reference_scale(A, p)
    threshold = BigFloat(0; precision = p)
    MA.operate_to!(threshold, *, relative, scale)
    threshold > absolute || MA.operate_to!(threshold, copy, absolute)
    return absolute, relative, scale, threshold
end

"""
    numerical_rank(F; atol=nothing, rtol=nothing) -> Int

Re-evaluate the RRQR numerical rank using the factor's recorded reference
scale.  Omitted tolerances reuse the factorization policy.  The factor's
packed storage and metadata are validated before reading them.
"""
function numerical_rank(
    F::BFLAQRFactor;
    atol::Union{Nothing,BigFloat}=nothing,
    rtol::Union{Nothing,BigFloat}=nothing,
)
    _validate_factor_integrity!(F, "numerical_rank")
    issuccess(F) || throw(ArgumentError(
        "numerical_rank: factor status is not successful",
    ))
    # The keyword tolerances, when provided, must carry the factor precision.
    kp = _check_precision(atol, rtol)
    if kp !== nothing && kp != F.precision_bits
        throw(PrecisionMismatch(F.precision_bits, kp, nothing))
    end
    a = atol === nothing ? F.atol : MA.mutable_copy(atol)
    r = rtol === nothing ? F.rtol : MA.mutable_copy(rtol)
    _qr_validate_tolerance(a, "atol", "numerical_rank")
    _qr_validate_tolerance(r, "rtol", "numerical_rank")
    threshold = BigFloat(0; precision = F.precision_bits)
    MA.operate_to!(threshold, *, r, F.reference_scale)
    threshold > a || MA.operate_to!(threshold, copy, a)
    return _qr_rank_from_factors(F.factors, threshold)
end

Base.size(F::BFLAQRFactor) = size(F.factors)
Base.size(F::BFLAQRFactor, dimension::Integer) = size(F.factors, dimension)
Base.eltype(::BFLAQRFactor{M,B}) where {M,B} = BigFloat

# --- public API ---------------------------------------------------------

"""
    qr!(backend, A; tol=nothing, atol=nothing, rtol=nothing) -> BFLAQRFactor

Rank-revealing column-pivoted QR factorization of `A` in place. The numerical
rank threshold is `max(atol, rtol * reference_scale)`, where
`reference_scale` is the largest input-column 2-norm. By default `atol` is zero
and `rtol` is `max(size(A)...)*eps(BigFloat)` at factor precision. The legacy
`tol` keyword remains an absolute-only mode and cannot be combined with
`atol` or `rtol`.
"""
function qr! end

function qr!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    tol::Union{Nothing,BigFloat}=nothing,
    atol::Union{Nothing,BigFloat}=nothing,
    rtol::Union{Nothing,BigFloat}=nothing,
)
    p = _require_precision(_check_precision(A, tol, atol, rtol), "qr!")
    _all_finite(A) || throw(DomainError(A, "qr!: input contains non-finite entries"))
    absolute, relative, scale, threshold = _qr_rank_policy(
        A, p, tol, atol, rtol, "qr!",
    )
    # Factor every numerically nonzero pivot. Rank policy is evaluated only
    # after the packed R exists, so callers may re-evaluate it later.
    zero_threshold = BigFloat(0; precision = p)
    tau, jpvt, _ = _qr!(backend, A, p, zero_threshold)
    (_all_finite(A) && _all_finite(tau)) || throw(DomainError(
        A, "qr!: factorization produced non-finite entries",
    ))
    rank = _qr_rank_from_factors(A, threshold)
    return BFLAQRFactor(
        A,
        backend,
        p,
        SUCCESS_STATUS,
        tau,
        jpvt,
        rank,
        MA.mutable_copy(absolute),
        absolute,
        relative,
        scale,
        threshold,
    )
end

"""
    qr(backend, A; tol=nothing, atol=nothing, rtol=nothing) -> BFLAQRFactor

Allocating column-pivoted QR factorization.
"""
function qr end

function qr(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    tol::Union{Nothing,BigFloat}=nothing,
    atol::Union{Nothing,BigFloat}=nothing,
    rtol::Union{Nothing,BigFloat}=nothing,
)
    p = _require_precision(_check_precision(A, tol, atol, rtol), "qr")
    return qr!(
        backend,
        owned_copy(A; precision_bits=p);
        tol=tol,
        atol=atol,
        rtol=rtol,
    )
end

"""
    applyQ!(F, B, trans=NoTrans) -> B

Apply `Q` (`NoTrans`) or `Qᵀ` (`Trans`) on the left, overwriting `B`
(an `m × k` matrix, where `m = size(F, 1)`).
"""
function applyQ!(F::BFLAQRFactor, B::AbstractVecOrMat{BigFloat}, trans::TransposeOp=NoTrans)
    issuccess(F) || throw(ArgumentError(
        "applyQ!: QR factor status is not successful",
    ))
    size(B, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("applyQ!: leading dimension differs"))
    _require_valid_transpose(trans, "applyQ!")
    _require_no_alias(B, F.factors, "applyQ!")
    _validate_factor_precision(F, "applyQ!", B)
    (_all_finite(F.factors) && _all_finite(F.tau) &&
     isfinite(F.tolerance) && isfinite(F.atol) && isfinite(F.rtol) &&
     isfinite(F.reference_scale) && isfinite(F.effective_threshold)) ||
        throw(DomainError(
        F, "applyQ!: factor storage contains non-finite entries",
    ))
    _all_finite(B) || throw(DomainError(
        B, "applyQ!: right-hand side contains non-finite entries",
    ))
    _apply_q!(F.backend, F, B, trans)
    _all_finite(B) || throw(DomainError(
        B, "applyQ!: operation produced non-finite entries",
    ))
    return B
end

_apply_q!(
    ::NativeBackend,
    F,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
) = _apply_q_common!(F, B, trans, workspace, workspace_worker)

_apply_q!(
    ::GenericBackend,
    F,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
) = _apply_q_common!(F, B, trans, workspace, workspace_worker)

function _apply_q_common!(
    F,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    p = F.precision_bits
    A = F.factors
    m = size(A, 1)
    r = length(F.tau)
    nrhs = length(B) ÷ m
    acc = _solve_scratch(workspace, workspace_worker, 1, p)
    buf = _solve_scratch(workspace, workspace_worker, 2, p)
    krange = trans === NoTrans ? (r:-1:1) : (1:r)
    @inbounds for k in krange
        tau = F.tau[k]
        for j in 1:nrhs
            base = (j - 1) * m
            MA.operate!(zero, acc)
            MA.operate_to!(acc, copy, B[base + k])  # v[1]=1 contributes B[k,j]
            for i in (k + 1):m
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], B[base + i])
            end
            # acc = vᵀ B[:,j]; B[k:m,j] -= tau*acc*v
            MA.operate_to!(acc, *, acc, tau)
            MA.operate_to!(buf, copy, acc)      # v[1]=1
            MA.operate_to!(B[base + k], -, B[base + k], buf)
            for i in (k + 1):m
                MA.operate_to!(buf, *, acc, A[i, k])
                MA.operate_to!(B[base + i], -, B[base + i], buf)
            end
        end
    end
    return B
end

function _qr_ldiv!(
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat},
    trusted::Bool,
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
    operation::AbstractString,
)
    issuccess(F) || throw(ArgumentError(
        "$operation: QR factor status is not successful",
    ))
    m, n = size(F.factors)
    size(rhs, 1) == m || throw(DimensionMismatch(
        "$operation: right-hand side rows differ",
    ))
    m >= n || throw(DimensionMismatch(
        "$operation: in-place QR solve requires rows >= columns",
    ))
    _require_no_alias(rhs, F.factors, operation)
    if trusted
        _validate_trusted_rhs_precision(F, operation, rhs)
    else
        _validate_factor_integrity!(F, operation)
        _validate_rhs_precision(F, operation, rhs)
    end
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: right-hand side contains non-finite entries",
    ))
    _validate_solve_workspace(
        workspace, workspace_worker, F.precision_bits, operation,
    )
    _qr_solve!(F.backend, F, rhs, workspace, workspace_worker)
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: solve produced non-finite entries",
    ))
    return rhs
end

function ldiv!(
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _qr_ldiv!(F, rhs, false, workspace, workspace_worker, "ldiv!")
end

function ldiv_trusted!(
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _qr_ldiv!(
        F, rhs, true, workspace, workspace_worker, "ldiv_trusted!",
    )
end

_qr_solve!(
    ::NativeBackend,
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
) = _qr_solve_common!(F, rhs, workspace, workspace_worker)

_qr_solve!(
    ::GenericBackend,
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
) = _qr_solve_common!(F, rhs, workspace, workspace_worker)

function _qr_solve_common!(
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
)
    m, n = size(F.factors)
    # y = Qᵀ rhs. This remains an explicit dispatch through the factor's
    # recorded backend even though Native and Generic share the arithmetic.
    _apply_q!(F.backend, F, rhs, Trans, workspace, workspace_worker)
    # Solve R[1:r,1:r] x1 = y[1:r]; x = P*x1. For a rank-deficient
    # overdetermined system, free variables in pivot order are set to zero.
    r = F.rank
    p = F.precision_bits
    A = F.factors
    acc = _solve_scratch(workspace, workspace_worker, 1, p)
    buf = _solve_scratch(workspace, workspace_worker, 2, p)
    storage = workspace === nothing ?
        owned_zeros(BigFloat, n; precision_bits=p) :
        workspace_buffer!(workspace, workspace_worker, n)
    x = view(storage, 1:n)
    nrhs = length(rhs) ÷ m
    @inbounds for col in 1:nrhs
        base = (col - 1) * m
        for i in 1:n
            MA.operate!(zero, x[i])
        end
        for i in r:-1:1
            MA.operate!(zero, acc)
            for k in (i + 1):r
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], x[k])
            end
            MA.operate_to!(acc, -, rhs[base + i], acc)
            _mpfr_div!(x[i], acc, A[i, i])
        end
        for i in 1:n
            MA.operate_to!(rhs[base + F.jpvt[i]], copy, x[i])
        end
    end
    return rhs
end

function solve!(
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return ldiv!(
        F, rhs; workspace=workspace, workspace_worker=workspace_worker,
    )
end

function solve(
    F::BFLAQRFactor,
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

# --- Native column-pivoted Householder QR -------------------------------

function _qr!(::NativeBackend, A::AbstractMatrix{BigFloat}, p::Int, tol::BigFloat)
    m, n = size(A)
    r = min(m, n)
    tau = [BigFloat(0; precision = p) for _ in 1:r]
    jpvt = collect(1:n)
    # Current and last exactly recomputed column 2-norms. Keeping both is the
    # LAPACK-style reliability signal for guarded downdates.
    column_norms = [BigFloat(0; precision = p) for _ in 1:n]
    exact_norms = [BigFloat(0; precision = p) for _ in 1:n]
    acc = BigFloat(0; precision = p)
    buf = BigFloat(0; precision = p)
    norm_scale = BigFloat(0; precision = p)
    norm_sumsq = BigFloat(0; precision = p)
    norm_absolute = BigFloat(0; precision = p)
    norm_ratio = BigFloat(0; precision = p)
    norm_term = BigFloat(0; precision = p)
    one_value = BigFloat(1; precision = p)
    two_value = BigFloat(2; precision = p)
    epsilon = BigFloat(0; precision = p)
    _mpfr_set_ui_2exp!(epsilon, 1, 1 - p)
    recompute_guard = BigFloat(0; precision = p)
    _mpfr_sqrt!(recompute_guard, epsilon)
    temporary = BigFloat(0; precision = p)
    temporary_2 = BigFloat(0; precision = p)
    s_scratch = BigFloat(0; precision = p)
    v1 = BigFloat(0; precision = p)
    for j in 1:n
        _qr_segment_norm!(
            column_norms[j], A, 1, j, norm_scale, norm_sumsq,
            norm_absolute, norm_ratio, norm_term, one_value,
        )
        MA.operate_to!(exact_norms[j], copy, column_norms[j])
    end
    rank = 0
    for k in 1:r
        piv = k
        mx = column_norms[k]
        for j in (k + 1):n
            if column_norms[j] > mx
                mx = column_norms[j]
                piv = j
            end
        end
        if mx <= tol
            break
        end
        if piv != k
            for i in 1:m
                A[i, k], A[i, piv] = A[i, piv], A[i, k]
            end
            jpvt[k], jpvt[piv] = jpvt[piv], jpvt[k]
            column_norms[k], column_norms[piv] =
                column_norms[piv], column_norms[k]
            exact_norms[k], exact_norms[piv] =
                exact_norms[piv], exact_norms[k]
        end
        # Householder reflector on A[k:m, k]
        _qr_segment_norm!(
            acc, A, k, k, norm_scale, norm_sumsq, norm_absolute,
            norm_ratio, norm_term, one_value,
        )
        x1 = A[k, k]
        if x1 >= 0
            MA.operate_to!(s_scratch, -, acc)     # s = -xnorm
        else
            MA.operate_to!(s_scratch, copy, acc)  # s = +xnorm
        end
        MA.operate_to!(v1, -, x1, s_scratch)      # v1 = x1 - s, at precision p
        MA.operate_to!(A[k, k], copy, s_scratch)  # R[k,k] = s
        # normalize v[k] = 1 implicitly; store v[i] = A[i,k]/v1 for i > k
        for i in (k + 1):m
            _mpfr_div!(A[i, k], A[i, k], v1)
        end
        # vnorm2 = v1² + Σ_{i>k} A[i,k]²
        MA.operate_to!(acc, copy, one_value)
        for i in (k + 1):m
            MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], A[i, k])
        end
        _mpfr_div!(tau[k], two_value, acc)
        # apply H to trailing columns
        for j in (k + 1):n
            MA.operate!(zero, acc)
            MA.operate_to!(acc, copy, A[k, j])  # v[1]=1 contributes A[k,j]
            for i in (k + 1):m
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], A[i, j])
            end
            MA.operate_to!(acc, *, acc, tau[k])
            MA.operate_to!(buf, copy, acc)      # v[1]=1
            MA.operate_to!(A[k, j], -, A[k, j], buf)
            for i in (k + 1):m
                MA.operate_to!(buf, *, acc, A[i, k])
                MA.operate_to!(A[i, j], -, A[i, j], buf)
            end
        end
        # LAPACK-style guarded downdate. When cancellation makes the estimate
        # unreliable, recompute the exact trailing norm in deterministic row
        # order before the next pivot selection.
        for j in (k + 1):n
            iszero(column_norms[j]) && continue
            _factor_abs_to!(norm_absolute, A[k, j])
            _mpfr_div!(temporary, norm_absolute, column_norms[j])
            if temporary >= one_value
                MA.operate!(zero, temporary_2)
            else
                MA.operate_to!(temporary_2, -, one_value, temporary)
                MA.operate_to!(norm_term, +, one_value, temporary)
                MA.operate!(*, temporary_2, norm_term)
            end
            if iszero(exact_norms[j])
                MA.operate!(zero, norm_term)
            else
                _mpfr_div!(norm_term, column_norms[j], exact_norms[j])
                MA.operate!(*, norm_term, norm_term)
                MA.operate!(*, norm_term, temporary_2)
            end
            if !isfinite(norm_term) || norm_term <= recompute_guard
                _qr_segment_norm!(
                    column_norms[j], A, k + 1, j, norm_scale,
                    norm_sumsq, norm_absolute, norm_ratio, norm_term,
                    one_value,
                )
                MA.operate_to!(exact_norms[j], copy, column_norms[j])
            else
                _mpfr_sqrt!(temporary_2, temporary_2)
                MA.operate!(*, column_norms[j], temporary_2)
            end
        end
        rank += 1
    end
    return tau, jpvt, rank
end

# --- Generic reference --------------------------------------------------

function _qr!(::GenericBackend, A::AbstractMatrix{BigFloat}, p::Int, tol::BigFloat)
    return _with_precision(p) do
        F = LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm())
        m, n = size(A)
        r = min(m, n)
        rank = 0
        for i in 1:r
            abs(F.factors[i, i]) > tol && (rank += 1)
        end
        # `qr` is allocating, so its packed factors do not live in `A`. Copy
        # them numerically into BFLA-owned storage before the temporary Julia
        # factor is discarded. The reflector scalars need the same treatment:
        # BigFloat array copies otherwise preserve mutable MPFR references.
        _convert_copy!(A, F.factors)
        tau = owned_zeros(BigFloat, r; precision_bits = p)
        _convert_copy!(tau, F.τ)
        return tau, collect(F.jpvt), rank
    end
end
