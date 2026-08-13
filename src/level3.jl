# BLAS Level 3 public API.

"""
    gemm!(backend, transA, transB, a, A, B, b, C) -> C

Matrix-matrix product `C = a * op(A) * op(B) + b * C`, where each `op` is the
identity or `transpose` per the corresponding `TransposeOp`. `C` must not alias
`A` or `B`.
"""
function gemm! end

function gemm!(
    backend::AbstractBFLABackend,
    transA::TransposeOp,
    transB::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    ;
    config::KernelConfig = KernelConfig(),
)
    _require_valid_transpose(transA, "gemm!")
    _require_valid_transpose(transB, "gemm!")
    _require_no_alias(C, A, "gemm!")
    _require_no_alias(C, B, "gemm!")
    mA, kA = size(A)
    mB, kB = size(B)
    m = transA === NoTrans ? mA : kA
    k = transA === NoTrans ? kA : mA
    k2 = transB === NoTrans ? mB : kB
    n = transB === NoTrans ? kB : mB
    k == k2 || throw(DimensionMismatch("gemm!: inner dimensions differ"))
    size(C) == (m, n) || throw(DimensionMismatch("gemm!: output dimensions differ"))
    p = _require_precision(_check_precision(a, b, A, B, C), "gemm!")
    return _gemm_dispatch!(backend, Val(transA), Val(transB), a, A, B, b, C, p, config)
end

"""
    syrk!(backend, triangle, trans, a, A, b, C) -> C

Rank-k symmetric update. For `NoTrans`, `C = a * A * transpose(A) + b * C`;
for `Trans`, `C = a * transpose(A) * A + b * C`. Only the requested triangle is
updated; the other triangle is left untouched. `C` must not alias `A`.
"""
function syrk! end

function syrk!(
    backend::AbstractBFLABackend,
    triangle::Triangle,
    trans::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    ;
    config::KernelConfig = KernelConfig(),
)
    _require_valid_triangle(triangle, "syrk!")
    _require_valid_transpose(trans, "syrk!")
    _require_no_alias(C, A, "syrk!")
    mA, kA = size(A)
    n = trans === NoTrans ? mA : kA
    size(C) == (n, n) || throw(DimensionMismatch("syrk!: output dimensions differ"))
    p = _require_precision(_check_precision(a, b, A, C), "syrk!")
    return _syrk_dispatch!(backend, triangle, Val(trans), a, A, b, C, p, config)
end

"""
    trmm!(backend, side, triangle, trans, diagonal, a, A, B) -> B

Triangular matrix product in place. `B = a * op(A) * B` for `LeftSide`, and
`B = a * B * op(A)` for `RightSide`. `B` must not alias `A`.
"""
function trmm! end

function trmm!(
    backend::AbstractBFLABackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
)
    _require_valid_side(side, "trmm!")
    _require_valid_triangle(triangle, "trmm!")
    _require_valid_transpose(trans, "trmm!")
    _require_valid_diagonal(diagonal, "trmm!")
    _require_square(A, "trmm!")
    _require_no_alias(B, A, "trmm!")
    n = size(A, 1)
    if side === LeftSide
        size(B, 1) == n || throw(DimensionMismatch("trmm!: left dimensions differ"))
    else
        size(B, 2) == n || throw(DimensionMismatch("trmm!: right dimensions differ"))
    end
    p = _require_precision(_check_precision(a, A, B), "trmm!")
    return _trmm!(backend, side, triangle, trans, diagonal, a, A, B, p)
end

"""
    trsm!(backend, side, triangle, trans, diagonal, a, A, B) -> B

Triangular solve in place. Solves `op(A) * X = a * B` for `LeftSide`, and
`X * op(A) = a * B` for `RightSide`, overwriting `B` with `X`. `B` must not
alias `A`.
"""
function trsm! end

function trsm!(
    backend::AbstractBFLABackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    ;
    config::KernelConfig = KernelConfig(),
)
    _require_valid_side(side, "trsm!")
    _require_valid_triangle(triangle, "trsm!")
    _require_valid_transpose(trans, "trsm!")
    _require_valid_diagonal(diagonal, "trsm!")
    _require_square(A, "trsm!")
    _require_no_alias(B, A, "trsm!")
    n = size(A, 1)
    if side === LeftSide
        size(B, 1) == n || throw(DimensionMismatch("trsm!: left dimensions differ"))
    else
        size(B, 2) == n || throw(DimensionMismatch("trsm!: right dimensions differ"))
    end
    p = _require_precision(_check_precision(a, A, B), "trsm!")
    return _trsm_dispatch!(backend, side, triangle, trans, diagonal, a, A, B, p, config)
end

"""
    gemmt!(backend, triangle, transA, transB, a, A, B, b, C) -> C

Symmetric matrix-matrix product `C = a * op(A) * op(B) + b * C` for the
requested authoritative triangle only; the other triangle is left untouched.
`C` is `n × n`, `op(A)` is `n × k`, and `op(B)` is `k × n`. `C` must not alias
`A` or `B`.
"""
function gemmt! end

function gemmt!(
    backend::AbstractBFLABackend,
    triangle::Triangle,
    transA::TransposeOp,
    transB::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
)
    _require_valid_triangle(triangle, "gemmt!")
    _require_valid_transpose(transA, "gemmt!")
    _require_valid_transpose(transB, "gemmt!")
    _require_no_alias(C, A, "gemmt!")
    _require_no_alias(C, B, "gemmt!")
    _require_square(C, "gemmt!")
    n = size(C, 1)
    rowsA = transA === NoTrans ? size(A, 1) : size(A, 2)
    columnsA = transA === NoTrans ? size(A, 2) : size(A, 1)
    rowsB = transB === NoTrans ? size(B, 1) : size(B, 2)
    columnsB = transB === NoTrans ? size(B, 2) : size(B, 1)
    rowsA == n || throw(DimensionMismatch(
        "gemmt!: rows of op(A) must match C",
    ))
    columnsB == n || throw(DimensionMismatch(
        "gemmt!: columns of op(B) must match C",
    ))
    columnsA == rowsB || throw(DimensionMismatch(
        "gemmt!: inner dimensions differ",
    ))
    p = _require_precision(_check_precision(a, b, A, B, C), "gemmt!")
    return _gemmt!(backend, triangle, Val(transA), Val(transB), a, A, B, b, C, p)
end

"""
    syr2k!(backend, triangle, trans, a, A, B, b, C) -> C

Rank-2k symmetric update. For `NoTrans`,
`C = a * (A * transpose(B) + B * transpose(A)) + b * C`; for `Trans`,
`C = a * (transpose(A) * B + transpose(B) * A) + b * C`. Only the requested
triangle is updated. `C` must not alias `A` or `B`.
"""
function syr2k! end

function syr2k!(
    backend::AbstractBFLABackend,
    triangle::Triangle,
    trans::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
)
    _require_valid_triangle(triangle, "syr2k!")
    _require_valid_transpose(trans, "syr2k!")
    _require_no_alias(C, A, "syr2k!")
    _require_no_alias(C, B, "syr2k!")
    _require_square(C, "syr2k!")
    n = size(C, 1)
    rowsA = trans === NoTrans ? size(A, 1) : size(A, 2)
    columnsA = trans === NoTrans ? size(A, 2) : size(A, 1)
    rowsB = trans === NoTrans ? size(B, 1) : size(B, 2)
    columnsB = trans === NoTrans ? size(B, 2) : size(B, 1)
    rowsA == n || throw(DimensionMismatch(
        "syr2k!: rows of op(A) must match C",
    ))
    rowsB == n || throw(DimensionMismatch(
        "syr2k!: rows of op(B) must match C",
    ))
    columnsA == columnsB || throw(DimensionMismatch(
        "syr2k!: contraction dimensions differ",
    ))
    p = _require_precision(_check_precision(a, b, A, B, C), "syr2k!")
    return _syr2k!(backend, triangle, Val(trans), a, A, B, b, C, p)
end
