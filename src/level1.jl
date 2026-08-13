# BLAS Level 1 public API. Each entry validates its arguments once and then
# delegates to a backend kernel; backend kernels never re-validate.

"""
    scal!(backend, a, x) -> x

In-place vector scale `x[i] = a * x[i]`.
"""
function scal! end

function scal!(backend::AbstractBFLABackend, a::BigFloat, x::AbstractVector{BigFloat})
    p = _require_precision(_check_precision(a, x), "scal!")
    return _scal!(backend, a, x, p)
end

"""
    axpy!(backend, a, x, y) -> y

In-place vector update `y = a * x + y`. `y` must not alias `x`.
"""
function axpy! end

function axpy!(backend::AbstractBFLABackend, a::BigFloat, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat})
    length(x) == length(y) ||
        throw(DimensionMismatch("axpy!: vector lengths differ"))
    _require_no_alias(y, x, "axpy!")
    p = _require_precision(_check_precision(a, x, y), "axpy!")
    return _axpy!(backend, a, x, y, p)
end

"""
    axpby!(backend, a, x, b, y) -> y

In-place vector update `y = a * x + b * y`. `y` must not alias `x`.
"""
function axpby! end

function axpby!(backend::AbstractBFLABackend, a::BigFloat, x::AbstractVector{BigFloat}, b::BigFloat, y::AbstractVector{BigFloat})
    length(x) == length(y) ||
        throw(DimensionMismatch("axpby!: vector lengths differ"))
    _require_no_alias(y, x, "axpby!")
    p = _require_precision(_check_precision(a, b, x, y), "axpby!")
    return _axpby!(backend, a, x, b, y, p)
end

"""
    dot(backend, x, y) -> BigFloat

Vector inner product `sum(x[i] * y[i])`.
"""
function dot end

function dot(backend::AbstractBFLABackend, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat})
    length(x) == length(y) ||
        throw(DimensionMismatch("dot: vector lengths differ"))
    p = _require_precision(_check_precision(x, y), "dot")
    return _dot(backend, x, y, p)
end

"""
    norminf(backend, x) -> BigFloat

Infinity norm `maximum(abs, x)`. Returns `NaN` if any element is `NaN`.

An empty `BigFloat` array has no elements from which to infer a precision, so
`norminf(backend, BigFloat[])` fails closed with an `ArgumentError` instead of
silently inheriting Julia's ambient `setprecision` context.
"""
function norminf end

function norminf(backend::AbstractBFLABackend, x::AbstractArray{BigFloat})
    isempty(x) && throw(ArgumentError(
        "norminf: cannot determine precision of an empty BigFloat array; " *
        "the infinity norm of an empty array is undefined in BFLA",
    ))
    p = _require_precision(_check_precision(x), "norminf")
    return _norminf(backend, x, p)
end
