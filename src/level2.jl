# BLAS Level 2 public API.

"""
    gemv!(backend, trans, a, A, x, b, y) -> y

Matrix-vector product `y = a * op(A) * x + b * y`, where `op(A)` is `A` for
`NoTrans` and `transpose(A)` for `Trans`. `y` must not alias `x`.
"""
function gemv! end

function gemv!(
    backend::AbstractBFLABackend,
    trans::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    b::BigFloat,
    y::AbstractVector{BigFloat},
)
    _require_valid_transpose(trans, "gemv!")
    _require_no_alias(y, x, "gemv!")
    m, n = size(A)
    if trans === NoTrans
        (length(x) == n && length(y) == m) ||
            throw(DimensionMismatch("gemv!: vector/matrix dimensions differ"))
    else
        (length(x) == m && length(y) == n) ||
            throw(DimensionMismatch("gemv!: vector/matrix dimensions differ"))
    end
    p = _require_precision(_check_precision(a, b, A, x, y), "gemv!")
    return _gemv!(backend, trans, a, A, x, b, y, p)
end

"""
    trsv!(backend, triangle, trans, diagonal, A, b) -> b

Solve the triangular system `op(A) * x = b` in place, overwriting `b` with
`x`. `op(A)` is `A` (`NoTrans`) or `transpose(A)` (`Trans`).
"""
function trsv! end

function trsv!(
    backend::AbstractBFLABackend,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    A::AbstractMatrix{BigFloat},
    b::AbstractVector{BigFloat},
)
    _require_valid_triangle(triangle, "trsv!")
    _require_valid_transpose(trans, "trsv!")
    _require_valid_diagonal(diagonal, "trsv!")
    _require_square(A, "trsv!")
    size(A, 1) == length(b) ||
        throw(DimensionMismatch("trsv!: right-hand side length differs"))
    _require_no_alias(b, A, "trsv!")
    p = _require_precision(_check_precision(A, b), "trsv!")
    return _trsv!(backend, triangle, trans, diagonal, A, b, p)
end

"""
    syr!(backend, triangle, a, x, A) -> A

Rank-one symmetric update `A = a * x * transpose(x) + A` for the entries in
the requested triangle only. The other triangle is left untouched.
"""
function syr! end

function syr!(
    backend::AbstractBFLABackend,
    triangle::Triangle,
    a::BigFloat,
    x::AbstractVector{BigFloat},
    A::AbstractMatrix{BigFloat},
)
    _require_valid_triangle(triangle, "syr!")
    _require_square(A, "syr!")
    size(A, 1) == length(x) ||
        throw(DimensionMismatch("syr!: vector/matrix dimensions differ"))
    _require_no_alias(A, x, "syr!")
    p = _require_precision(_check_precision(a, x, A), "syr!")
    return _syr!(backend, triangle, a, x, A, p)
end
