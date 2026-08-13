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
    tolerance::BigFloat
end

factor_matrix(F::BFLAQRFactor) = F.factors
factor_backend(F::BFLAQRFactor) = F.backend
factor_precision(F::BFLAQRFactor) = F.precision_bits
factor_status(F::BFLAQRFactor) = F.status
factor_kind(::BFLAQRFactor) = :qr
issuccess(F::BFLAQRFactor) = F.status.kind === :success

"""
    factor_rank(F) -> Int

Numerical rank of the factorization.
"""
factor_rank(F::BFLAQRFactor) = F.rank

"""
    factor_jpvt(F) -> Vector{Int}

Column permutation (`A * P = Q * R`).
"""
factor_jpvt(F::BFLAQRFactor) = copy(F.jpvt)

"""
    factor_Rdiag(F) -> Vector{BigFloat}

Diagonal entries of the `R` factor, in pivot order.
"""
function factor_Rdiag(F::BFLAQRFactor)
    return [
        MA.mutable_copy(F.factors[i, i])
        for i in 1:min(size(F.factors, 1), size(F.factors, 2))
    ]
end

factor_tolerance(F::BFLAQRFactor) = MA.mutable_copy(F.tolerance)

Base.size(F::BFLAQRFactor) = size(F.factors)
Base.size(F::BFLAQRFactor, dimension::Integer) = size(F.factors, dimension)
Base.eltype(::BFLAQRFactor{M,B}) where {M,B} = BigFloat

# --- public API ---------------------------------------------------------

"""
    qr!(backend, A; tol=nothing) -> BFLAQRFactor

Rank-revealing column-pivoted QR factorization of `A` in place. `tol` is the
absolute tolerance on the `R` diagonal used to determine the numerical rank;
`nothing` is equivalent to an exact-zero tolerance.
"""
function qr! end

function qr!(backend::AbstractBFLABackend, A::AbstractMatrix{BigFloat}; tol::Union{Nothing,BigFloat}=nothing)
    p = _require_precision(_check_precision(A, tol), "qr!")
    _all_finite(A) || throw(DomainError(A, "qr!: input contains non-finite entries"))
    m, n = size(A)
    tolval = tol === nothing ? BigFloat(0; precision = p) : MA.mutable_copy(tol)
    isfinite(tolval) || throw(DomainError(tolval, "qr!: tolerance must be finite"))
    tolval >= 0 || throw(DomainError(tolval, "qr!: tolerance must be nonnegative"))
    tau, jpvt, rank = _qr!(backend, A, p, tolval)
    (_all_finite(A) && _all_finite(tau)) || throw(DomainError(
        A, "qr!: factorization produced non-finite entries",
    ))
    return BFLAQRFactor(A, backend, p, SUCCESS_STATUS, tau, jpvt, rank, tolval)
end

"""
    qr(backend, A; tol=nothing) -> BFLAQRFactor

Allocating column-pivoted QR factorization.
"""
function qr end

function qr(backend::AbstractBFLABackend, A::AbstractMatrix{BigFloat}; tol::Union{Nothing,BigFloat}=nothing)
    p = _require_precision(_check_precision(A), "qr")
    return qr!(backend, owned_copy(A; precision_bits=p); tol=tol)
end

"""
    applyQ!(F, B, trans=NoTrans) -> B

Apply `Q` (`NoTrans`) or `Qᵀ` (`Trans`) on the left, overwriting `B`
(an `m × k` matrix, where `m = size(F, 1)`).
"""
function applyQ!(F::BFLAQRFactor, B::AbstractVecOrMat{BigFloat}, trans::TransposeOp=NoTrans)
    size(B, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("applyQ!: leading dimension differs"))
    _require_valid_transpose(trans, "applyQ!")
    _require_no_alias(B, F.factors, "applyQ!")
    p_actual = _require_precision(
        _check_precision(F.factors, F.tau, F.tolerance, B), "applyQ!",
    )
    p_actual == F.precision_bits ||
        throw(PrecisionMismatch(F.precision_bits, p_actual, nothing))
    (_all_finite(F.factors) && _all_finite(F.tau) &&
     isfinite(F.tolerance)) || throw(DomainError(
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
    F::BFLAQRFactor,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
) = _apply_q_common!(F, B, trans)

_apply_q!(
    ::GenericBackend,
    F::BFLAQRFactor,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
) = _apply_q_common!(F, B, trans)

function _apply_q_common!(
    F::BFLAQRFactor,
    B::AbstractVecOrMat{BigFloat},
    trans::TransposeOp,
)
    p = F.precision_bits
    A = F.factors
    m = size(A, 1)
    r = length(F.tau)
    R = reshape(B, m, :)
    acc = BigFloat(0; precision = p)
    buf = BigFloat(0; precision = p)
    krange = trans === NoTrans ? (r:-1:1) : (1:r)
    @inbounds for k in krange
        tau = F.tau[k]
        for j in axes(R, 2)
            MA.operate!(zero, acc)
            MA.operate_to!(acc, copy, R[k, j])  # v[1]=1 contributes B[k,j]
            for i in (k + 1):m
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], R[i, j])
            end
            # acc = vᵀ B[:,j]; B[k:m,j] -= tau*acc*v
            MA.operate_to!(acc, *, acc, tau)
            MA.operate_to!(buf, copy, acc)      # v[1]=1
            MA.operate_to!(R[k, j], -, R[k, j], buf)
            for i in (k + 1):m
                MA.operate_to!(buf, *, acc, A[i, k])
                MA.operate_to!(R[i, j], -, R[i, j], buf)
            end
        end
    end
    return B
end

function ldiv!(F::BFLAQRFactor, rhs::AbstractVecOrMat{BigFloat})
    m, n = size(F.factors)
    size(rhs, 1) == m || throw(DimensionMismatch("ldiv!: right-hand side rows differ"))
    m >= n || throw(DimensionMismatch(
        "ldiv!: in-place QR solve requires rows >= columns",
    ))
    _require_no_alias(rhs, F.factors, "ldiv!")
    p_actual = _require_precision(
        _check_precision(F.factors, F.tau, F.tolerance, rhs), "ldiv!",
    )
    p_actual == F.precision_bits || throw(PrecisionMismatch(F.precision_bits, p_actual, nothing))
    (_all_finite(F.factors) && _all_finite(F.tau) &&
     isfinite(F.tolerance)) || throw(DomainError(
        F, "ldiv!: factor storage contains non-finite entries",
    ))
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: right-hand side contains non-finite entries",
    ))
    _qr_solve!(F.backend, F, rhs)
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: solve produced non-finite entries",
    ))
    return rhs
end

_qr_solve!(
    ::NativeBackend,
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat},
) = _qr_solve_common!(F, rhs)

_qr_solve!(
    ::GenericBackend,
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat},
) = _qr_solve_common!(F, rhs)

function _qr_solve_common!(
    F::BFLAQRFactor,
    rhs::AbstractVecOrMat{BigFloat},
)
    m, n = size(F.factors)
    # y = Qᵀ rhs. This remains an explicit dispatch through the factor's
    # recorded backend even though Native and Generic share the arithmetic.
    _apply_q!(F.backend, F, rhs, Trans)
    # Solve R[1:r,1:r] x1 = y[1:r]; x = P*x1. For a rank-deficient
    # overdetermined system, free variables in pivot order are set to zero.
    r = F.rank
    p = F.precision_bits
    A = F.factors
    acc = BigFloat(0; precision = p)
    buf = BigFloat(0; precision = p)
    R = reshape(rhs, m, :)
    @inbounds for col in axes(R, 2)
        x = [BigFloat(0; precision = p) for _ in 1:n]
        for i in r:-1:1
            MA.operate!(zero, acc)
            for k in (i + 1):r
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], x[k])
            end
            MA.operate_to!(acc, -, R[i, col], acc)
            _mpfr_div!(x[i], acc, A[i, i])
        end
        for i in 1:n
            MA.operate_to!(R[F.jpvt[i], col], copy, x[i])
        end
    end
    return rhs
end

solve!(F::BFLAQRFactor, rhs::AbstractVecOrMat{BigFloat}) = ldiv!(F, rhs)
solve(F::BFLAQRFactor, rhs::AbstractVecOrMat{BigFloat}) = ldiv!(F, owned_copy(rhs))

# --- Native column-pivoted Householder QR -------------------------------

function _qr!(::NativeBackend, A::AbstractMatrix{BigFloat}, p::Int, tol::BigFloat)
    m, n = size(A)
    r = min(m, n)
    tau = [BigFloat(0; precision = p) for _ in 1:r]
    jpvt = collect(1:n)
    tol_sq = BigFloat(0; precision = p)
    MA.operate_to!(tol_sq, *, tol, tol)
    # squared column norms
    cn = [BigFloat(0; precision = p) for _ in 1:n]
    acc = BigFloat(0; precision = p)
    buf = BigFloat(0; precision = p)
    for j in 1:n
        MA.operate!(zero, acc)
        for i in 1:m
            MA.buffered_operate!(buf, MA.add_mul, acc, A[i, j], A[i, j])
        end
        MA.operate_to!(cn[j], copy, acc)
    end
    rank = 0
    for k in 1:r
        piv = k
        mx = cn[k]
        for j in (k + 1):n
            if cn[j] > mx
                mx = cn[j]
                piv = j
            end
        end
        if mx <= tol_sq
            break
        end
        if piv != k
            for i in 1:m
                A[i, k], A[i, piv] = A[i, piv], A[i, k]
            end
            jpvt[k], jpvt[piv] = jpvt[piv], jpvt[k]
            cn[k], cn[piv] = cn[piv], cn[k]
        end
        # Householder reflector on A[k:m, k]
        MA.operate!(zero, acc)
        for i in k:m
            MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], A[i, k])
        end
        _mpfr_sqrt!(acc, acc)  # xnorm
        x1 = A[k, k]
        s_scratch = BigFloat(0; precision = p)
        if x1 >= 0
            MA.operate_to!(s_scratch, -, acc)     # s = -xnorm
        else
            MA.operate_to!(s_scratch, copy, acc)  # s = +xnorm
        end
        v1 = BigFloat(0; precision = p)
        MA.operate_to!(v1, -, x1, s_scratch)      # v1 = x1 - s, at precision p
        MA.operate_to!(A[k, k], copy, s_scratch)  # R[k,k] = s
        # normalize v[k] = 1 implicitly; store v[i] = A[i,k]/v1 for i > k
        for i in (k + 1):m
            _mpfr_div!(A[i, k], A[i, k], v1)
        end
        # vnorm2 = v1² + Σ_{i>k} A[i,k]²
        MA.operate_to!(acc, copy, BigFloat(1; precision = p))
        for i in (k + 1):m
            MA.buffered_operate!(buf, MA.add_mul, acc, A[i, k], A[i, k])
        end
        _mpfr_div!(tau[k], BigFloat(2; precision = p), acc)
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
        # update column norms: cn[j] -= A[k,j]² for j > k
        for j in (k + 1):n
            MA.operate_to!(buf, *, A[k, j], A[k, j])
            MA.operate_to!(cn[j], -, cn[j], buf)
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
