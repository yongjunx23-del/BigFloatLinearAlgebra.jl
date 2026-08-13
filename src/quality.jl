# Solver-independent residual and numerical-quality primitives.

"""
    residual!(backend, trans, A, x, b, residual; config) -> residual
    residual!(backend, A, x, b, residual; config) -> residual

Compute `residual = b - op(A) * x` in caller-owned storage. `x`, `b`, and
`residual` may be vectors or matrices (multiple right-hand sides), but must
have the same dimensionality. All operands use one uniform precision and
`residual` must not alias an input.
"""
function residual! end

function residual!(
    backend::AbstractBFLABackend,
    trans::TransposeOp,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat};
    config::KernelConfig=KernelConfig(),
)
    _validate_residual_dimensions(trans, A, x, b, residual, "residual!")
    _require_no_alias(residual, A, "residual!")
    _require_no_alias(residual, x, "residual!")
    _require_no_alias(residual, b, "residual!")
    p = _require_precision(
        _check_precision(A, x, b, residual),
        "residual!",
    )
    (_all_finite(A) && _all_finite(x) && _all_finite(b)) ||
        throw(DomainError((A, x, b), "residual!: inputs must be finite"))
    return _residual!(backend, trans, A, x, b, residual, p, config)
end

residual!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat};
    kwargs...,
) = residual!(backend, NoTrans, A, x, b, residual; kwargs...)

function _validate_residual_dimensions(trans, A, x, b, residual, operation)
    _require_valid_transpose(trans, operation)
    ndims(x) == ndims(b) == ndims(residual) || throw(DimensionMismatch(
        "$operation: x, b, and residual must all be vectors or all be matrices",
    ))
    mA, nA = size(A)
    m = trans === NoTrans ? mA : nA
    n = trans === NoTrans ? nA : mA
    size(x, 1) == n || throw(DimensionMismatch(
        "$operation: solution rows differ from op(A) columns",
    ))
    size(b, 1) == m || throw(DimensionMismatch(
        "$operation: right-hand side rows differ from op(A) rows",
    ))
    size(residual) == size(b) || throw(DimensionMismatch(
        "$operation: residual and right-hand side dimensions differ",
    ))
    if ndims(x) == 2
        size(x, 2) == size(b, 2) || throw(DimensionMismatch(
            "$operation: right-hand side column counts differ",
        ))
    end
    isempty(residual) && throw(ArgumentError(
        "$operation: empty residuals have no defined normwise quality metric",
    ))
    return nothing
end

function _residual!(
    backend::Union{NativeBackend,GenericBackend},
    trans,
    A,
    x,
    b,
    residual,
    p,
    config,
)
    _convert_copy!(residual, b)
    minus_one = BigFloat(-1; precision = p)
    one_value = BigFloat(1; precision = p)
    if x isa AbstractVector
        return _gemv!(backend, trans, minus_one, A, x, one_value, residual, p)
    end
    return _gemm_dispatch!(
        backend,
        Val(trans),
        Val(NoTrans),
        minus_one,
        A,
        x,
        one_value,
        residual,
        p,
        config,
    )
end

"""
    normwise_backward_error(backend, trans, A, x, b, residual) -> BigFloat

Evaluate

`norm(residual, Inf) / (norm(op(A), Inf) * norm(x, Inf) + norm(b, Inf))`

from a caller-supplied residual. Matrix arguments use the induced infinity
norm (maximum absolute row sum). This function reports a number only; it does
not decide whether the result is acceptable. If both numerator and denominator
are zero it returns zero; a nonzero numerator over a zero denominator returns
`Inf` at the operand precision.
"""
function normwise_backward_error end

function normwise_backward_error(
    backend::AbstractBFLABackend,
    trans::TransposeOp,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat},
)
    _validate_residual_dimensions(
        trans, A, x, b, residual, "normwise_backward_error",
    )
    p = _require_precision(
        _check_precision(A, x, b, residual),
        "normwise_backward_error",
    )
    (_all_finite(A) && _all_finite(x) && _all_finite(b) &&
     _all_finite(residual)) || throw(DomainError(
        (A, x, b, residual),
        "normwise_backward_error: inputs must be finite",
    ))
    return _normwise_backward_error(backend, trans, A, x, b, residual, p)
end

normwise_backward_error(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat},
) = normwise_backward_error(backend, NoTrans, A, x, b, residual)

"""
    normwise_backward_error(backend, trans, A, x, b; config)

Allocating convenience form. It creates ownership-safe residual storage,
computes the residual, and returns the normwise backward error. Repeated callers
should provide their own residual to [`residual!`](@ref) and use the six-argument
form to avoid the allocation.
"""
function normwise_backward_error(
    backend::AbstractBFLABackend,
    trans::TransposeOp,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat};
    config::KernelConfig=KernelConfig(),
)
    residual = owned_similar(b)
    residual!(backend, trans, A, x, b, residual; config=config)
    return normwise_backward_error(backend, trans, A, x, b, residual)
end

normwise_backward_error(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat};
    kwargs...,
) = normwise_backward_error(backend, NoTrans, A, x, b; kwargs...)

function _normwise_backward_error(
    ::Union{NativeBackend,GenericBackend},
    trans,
    A,
    x,
    b,
    residual,
    p,
)
    return _normwise_backward_error_at_precision(trans, A, x, b, residual, p)
end

function _abs_to!(destination::BigFloat, source::BigFloat)
    MA.operate_to!(destination, copy, source)
    signbit(destination) && MA.operate!(-, destination)
    return destination
end

function _matrix_op_norminf(A, trans, p)
    mA, nA = size(A)
    rows = trans === NoTrans ? mA : nA
    columns = trans === NoTrans ? nA : mA
    maximum_value = BigFloat(0; precision = p)
    row_sum = BigFloat(0; precision = p)
    absolute_value = BigFloat(0; precision = p)
    @inbounds for i in 1:rows
        MA.operate!(zero, row_sum)
        for j in 1:columns
            value = trans === NoTrans ? A[i, j] : A[j, i]
            _abs_to!(absolute_value, value)
            MA.operate!(+, row_sum, absolute_value)
        end
        row_sum > maximum_value && MA.operate_to!(maximum_value, copy, row_sum)
    end
    return maximum_value
end

function _rhs_norminf(X::AbstractVector{BigFloat}, p)
    maximum_value = BigFloat(0; precision = p)
    absolute_value = BigFloat(0; precision = p)
    @inbounds for value in X
        _abs_to!(absolute_value, value)
        absolute_value > maximum_value &&
            MA.operate_to!(maximum_value, copy, absolute_value)
    end
    return maximum_value
end

function _rhs_norminf(X::AbstractMatrix{BigFloat}, p)
    maximum_value = BigFloat(0; precision = p)
    row_sum = BigFloat(0; precision = p)
    absolute_value = BigFloat(0; precision = p)
    @inbounds for i in axes(X, 1)
        MA.operate!(zero, row_sum)
        for j in axes(X, 2)
            _abs_to!(absolute_value, X[i, j])
            MA.operate!(+, row_sum, absolute_value)
        end
        row_sum > maximum_value && MA.operate_to!(maximum_value, copy, row_sum)
    end
    return maximum_value
end

function _normwise_backward_error_at_precision(trans, A, x, b, residual, p)
    norm_A = _matrix_op_norminf(A, trans, p)
    norm_x = _rhs_norminf(x, p)
    norm_b = _rhs_norminf(b, p)
    norm_r = _rhs_norminf(residual, p)
    product = BigFloat(0; precision = p)
    denominator = BigFloat(0; precision = p)
    result = BigFloat(0; precision = p)
    MA.operate_to!(product, *, norm_A, norm_x)
    MA.operate_to!(denominator, +, product, norm_b)
    if iszero(denominator)
        return iszero(norm_r) ? result : BigFloat(Inf; precision = p)
    end
    _mpfr_div!(result, norm_r, denominator)
    return result
end

"""
    higher_precision_residual!(
        backend, trans, A, x, b, residual;
        residual_precision,
        factor_precision=nothing,
    ) -> NamedTuple

Compute `residual = b - op(A)*x` at the explicitly higher precision carried by
the caller-owned `residual` storage. `A`, `x`, and `b` must share one uniform
factor/solve precision `p`; every residual entry must have precision `q > p`.
Low-precision values are numerically promoted into q-bit arithmetic without
changing the ordinary same-precision kernel contract.

The return value contains `residual`, `factor_precision`,
`residual_precision`, and the q-bit `backward_error`. These are diagnostics
only: BFLA does not accept/reject the result, refine, increase precision, or
switch backend automatically.
"""
function higher_precision_residual! end

function higher_precision_residual!(
    backend::AbstractBFLABackend,
    trans::TransposeOp,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat};
    residual_precision::Int,
    factor_precision::Union{Nothing,Int}=nothing,
)
    _validate_residual_dimensions(
        trans, A, x, b, residual, "higher_precision_residual!",
    )
    _require_no_alias(residual, A, "higher_precision_residual!")
    _require_no_alias(residual, x, "higher_precision_residual!")
    _require_no_alias(residual, b, "higher_precision_residual!")
    p = _require_precision(
        _check_precision(A, x, b),
        "higher_precision_residual!",
    )
    factor_precision === nothing || factor_precision == p ||
        throw(PrecisionMismatch(factor_precision, p, nothing))
    q_actual = _require_precision(
        _check_precision(residual),
        "higher_precision_residual!",
    )
    residual_precision == q_actual ||
        throw(PrecisionMismatch(residual_precision, q_actual, nothing))
    q = residual_precision
    q > p || throw(ArgumentError(
        "higher_precision_residual!: residual precision ($q) must exceed " *
        "factor precision ($p)",
    ))
    (_all_finite(A) && _all_finite(x) && _all_finite(b)) ||
        throw(DomainError(
            (A, x, b),
            "higher_precision_residual!: inputs must be finite",
        ))
    _higher_precision_residual!(backend, trans, A, x, b, residual, q)
    _all_finite(residual) || throw(DomainError(
        residual,
        "higher_precision_residual!: computation produced a non-finite residual",
    ))
    backward_error = _normwise_backward_error_at_precision(
        trans, A, x, b, residual, q,
    )
    return (
        residual = residual,
        factor_precision = p,
        residual_precision = q,
        backward_error = backward_error,
    )
end

higher_precision_residual!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat};
    kwargs...,
) = higher_precision_residual!(backend, NoTrans, A, x, b, residual; kwargs...)

function _higher_precision_residual!(
    ::NativeBackend,
    trans,
    A,
    x,
    b,
    residual,
    q,
)
    _convert_copy!(residual, b)
    accumulator = BigFloat(0; precision = q)
    buffer = BigFloat(0; precision = q)
    X = reshape(x, size(x, 1), :)
    R = reshape(residual, size(residual, 1), :)
    inner = size(X, 1)
    @inbounds for column in axes(R, 2), row in axes(R, 1)
        MA.operate!(zero, accumulator)
        for k in 1:inner
            value = trans === NoTrans ? A[row, k] : A[k, row]
            MA.buffered_operate!(
                buffer, MA.add_mul, accumulator, value, X[k, column],
            )
        end
        MA.operate_to!(R[row, column], -, R[row, column], accumulator)
    end
    return residual
end

function _higher_precision_residual!(
    ::GenericBackend,
    trans,
    A,
    x,
    b,
    residual,
    q,
)
    return _with_precision(q) do
        X = reshape(x, size(x, 1), :)
        B = reshape(b, size(b, 1), :)
        R = reshape(residual, size(residual, 1), :)
        inner = size(X, 1)
        @inbounds for column in axes(R, 2), row in axes(R, 1)
            accumulator = zero(BigFloat)
            for k in 1:inner
                value = trans === NoTrans ? A[row, k] : A[k, row]
                accumulator += value * X[k, column]
            end
            R[row, column] = B[row, column] - accumulator
        end
        residual
    end
end

"""
    refine_once!(factor, A, x, b, residual, correction; trans=NoTrans)

Perform exactly one iterative-refinement correction using the backend and
precision recorded by `factor`:

1. `residual = b - op(A)*x` (at the precision of `residual`),
2. explicitly round `residual` into factor-precision `correction`,
3. solve the factor system for `correction`, and
4. update `x += correction`.

`A`, `x`, `b`, `correction`, and the factor storage must all use factor
precision `p`. `residual` may use `p` or an explicitly higher precision `q`.
The operation returns diagnostics before and after the one step; it does not
iterate, choose a tolerance, increase precision, change factorization, or
switch backend. Validation completes before output is written, but after the
step starts `residual` and `correction` may be overwritten if a numerical solve
fails; no rollback is promised. `x` is updated only after the solve succeeds.
"""
function refine_once! end

function refine_once!(
    factor::AbstractBFLAFactor,
    A::AbstractMatrix{BigFloat},
    x::AbstractVecOrMat{BigFloat},
    b::AbstractVecOrMat{BigFloat},
    residual::AbstractVecOrMat{BigFloat},
    correction::AbstractVecOrMat{BigFloat};
    trans::TransposeOp=NoTrans,
)
    _require_valid_transpose(trans, "refine_once!")
    trans === NoTrans || throw(UnsupportedOperation(
        factor_backend(factor),
        :refine_once!,
        "transpose refinement requires a transpose factor solve",
    ))
    _require_square(A, "refine_once!")
    factors = factor_matrix(factor)
    size(factors, 1) == size(factors, 2) == size(A, 1) ||
        throw(DimensionMismatch(
            "refine_once!: factor and coefficient matrix dimensions differ",
        ))
    _validate_residual_dimensions(
        trans, A, x, b, residual, "refine_once!",
    )
    size(correction) == size(x) || throw(DimensionMismatch(
        "refine_once!: correction and solution dimensions differ",
    ))
    ndims(correction) == ndims(x) || throw(DimensionMismatch(
        "refine_once!: correction and solution dimensionality differ",
    ))
    issuccess(factor) || throw(ArgumentError(
        "refine_once!: factor status is not successful",
    ))
    factor isa BFLAQRFactor && factor_rank(factor) < size(A, 1) &&
        throw(LinearAlgebra.SingularException(factor_rank(factor) + 1))

    for (destination, source) in (
        (x, A),
        (x, b),
        (x, residual),
        (x, correction),
        (x, factors),
        (residual, A),
        (residual, b),
        (residual, correction),
        (residual, factors),
        (correction, A),
        (correction, b),
        (correction, factors),
        (A, factors),
    )
        _require_no_alias(destination, source, "refine_once!")
    end

    p = _require_precision(
        _check_precision(factors, A, x, b, correction),
        "refine_once!",
    )
    p == factor_precision(factor) ||
        throw(PrecisionMismatch(factor_precision(factor), p, nothing))
    q = _require_precision(_check_precision(residual), "refine_once!")
    q >= p || throw(ArgumentError(
        "refine_once!: residual precision ($q) must be at least factor " *
        "precision ($p)",
    ))
    (_factor_storage_finite(factor) && _all_finite(A) && _all_finite(x) &&
     _all_finite(b)) || throw(DomainError(
        (factor, A, x, b), "refine_once!: inputs must be finite",
    ))

    backend = factor_backend(factor)
    before = if q == p
        residual!(backend, trans, A, x, b, residual)
        normwise_backward_error(backend, trans, A, x, b, residual)
    else
        report = higher_precision_residual!(
            backend,
            trans,
            A,
            x,
            b,
            residual;
            residual_precision=q,
            factor_precision=p,
        )
        report.backward_error
    end

    convert_owned!(correction, residual)
    solve!(factor, correction)
    _refinement_update!(backend, x, correction, p)

    after = if q == p
        residual!(backend, trans, A, x, b, residual)
        normwise_backward_error(backend, trans, A, x, b, residual)
    else
        report = higher_precision_residual!(
            backend,
            trans,
            A,
            x,
            b,
            residual;
            residual_precision=q,
            factor_precision=p,
        )
        report.backward_error
    end

    return (
        x = x,
        residual = residual,
        correction = correction,
        backend = backend,
        factor_precision = p,
        residual_precision = q,
        backward_error_before = before,
        backward_error_after = after,
    )
end

_factor_storage_finite(F::BFLACholeskyFactor) =
    _triangle_finite(F.factors, F.triangle)
_factor_storage_finite(F::BFLALDLTFactor) =
    _triangle_finite(F.factors, Lower)
_factor_storage_finite(F::BFLAQRFactor) = _all_finite(F.factors)
_factor_storage_finite(F::BFLALUFactor) = _all_finite(F.factors)

function _refinement_update!(backend, x, correction, p)
    one_value = BigFloat(1; precision = p)
    X = reshape(x, length(x))
    D = reshape(correction, length(correction))
    return _axpy!(backend, one_value, D, X, p)
end
