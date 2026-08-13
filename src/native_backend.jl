# NativeBackend kernels, extracted and generalized from the SDPX legacy BigFloat
# dense kernels (src/kernels/bigfloat.jl). See THIRD_PARTY_NOTICES.md for the
# provenance and license.
#
# The extraction preserves the original reduction order and MPFR ownership
# discipline: dot products accumulate into scratch through MutableArithmetics'
# `buffered_operate!` (a correctly rounded `mpfr_mul` followed by `mpfr_add`,
# matching the frozen legacy trajectory), square roots and divisions go through
# the `mpfr_sqrt`/`mpfr_div` wrappers in mpfr.jl, and every destination is a
# `BigFloat` created at an explicit target precision rather than the ambient
# global precision.
#
# In-place kernels (`gemv!`, `gemm!`, `syrk!`, `syr!`, `trsv!`, `trsm!`,
# `trmm!`, `scal!`, `axpy!`, `axpby!`) write directly into their destination
# storage. The destination must therefore be independently owned (for example,
# produced by `owned_zeros`/`owned_similar`/`owned_copy`); aliased storage such
# as `zeros(BigFloat, ...)` is not a valid destination. Inter-operand aliasing
# is rejected by the public validation layer.

@inline _scratch(p::Int) = BigFloat(0; precision = p)

@inline function _update_abs_max!(maximum_value::BigFloat, negative_maximum::BigFloat, value::BigFloat)
    if signbit(value)
        if value < negative_maximum
            MA.operate_to!(negative_maximum, copy, value)
            MA.operate_to!(maximum_value, -, value)
        end
    elseif value > maximum_value
        MA.operate_to!(maximum_value, copy, value)
        MA.operate_to!(negative_maximum, -, value)
    end
    return nothing
end

@inline function _store_owned!(
    destination::BigFloat,
    accumulator::BigFloat,
    buffer::BigFloat,
    a::BigFloat,
    b::BigFloat,
    alpha_is_one::Bool,
    beta_is_zero::Bool,
)
    if !beta_is_zero
        MA.operate_to!(buffer, *, b, destination)
    end
    if alpha_is_one
        MA.operate_to!(destination, copy, accumulator)
    else
        MA.operate_to!(destination, *, a, accumulator)
    end
    if !beta_is_zero
        MA.operate!(+, destination, buffer)
    end
    return nothing
end

function _native_scale!(B::AbstractArray{BigFloat}, a::BigFloat)
    isone(a) && return B
    @inbounds for index in eachindex(B)
        MA.operate_to!(B[index], *, a, B[index])
    end
    return B
end

# Level 1 ----------------------------------------------------------------

function _scal!(::NativeBackend, a::BigFloat, x::AbstractVector{BigFloat}, p::Int)
    isone(a) && return x
    @inbounds for index in eachindex(x)
        MA.operate_to!(x[index], *, a, x[index])
    end
    return x
end

function _axpy!(::NativeBackend, a::BigFloat, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat}, p::Int)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    @inbounds for index in eachindex(x, y)
        if alpha_is_one
            MA.operate!(+, y[index], x[index])
        else
            MA.operate_to!(buffer, *, a, x[index])
            MA.operate!(+, y[index], buffer)
        end
    end
    return y
end

function _axpby!(::NativeBackend, a::BigFloat, x::AbstractVector{BigFloat}, b::BigFloat, y::AbstractVector{BigFloat}, p::Int)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for index in eachindex(x, y)
        if !beta_is_zero
            MA.operate_to!(buffer, *, b, y[index])
        end
        if alpha_is_one
            MA.operate_to!(y[index], copy, x[index])
        else
            MA.operate_to!(y[index], *, a, x[index])
        end
        if !beta_is_zero
            MA.operate!(+, y[index], buffer)
        end
    end
    return y
end

function _dot(::NativeBackend, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat}, p::Int)
    acc = _scratch(p)
    buffer = _scratch(p)
    MA.operate!(zero, acc)
    @inbounds for index in eachindex(x, y)
        MA.buffered_operate!(buffer, MA.add_mul, acc, x[index], y[index])
    end
    return acc
end

function _norminf(::NativeBackend, x::AbstractArray{BigFloat}, p::Int)
    maximum_value = _scratch(p)
    negative_maximum = _scratch(p)
    MA.operate!(zero, maximum_value)
    MA.operate!(zero, negative_maximum)
    @inbounds for value in x
        if isnan(value)
            return MA.mutable_copy(value)
        end
        _update_abs_max!(maximum_value, negative_maximum, value)
    end
    return maximum_value
end

# Reduction helpers ------------------------------------------------------

@inline function _dot_row_col!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{NoTrans},
    ::Val{NoTrans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], B[l, j])
    end
    return acc
end

@inline function _dot_row_col!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{NoTrans},
    ::Val{Trans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], B[j, l])
    end
    return acc
end

@inline function _dot_row_col!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{Trans},
    ::Val{NoTrans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], B[l, j])
    end
    return acc
end

@inline function _dot_row_col!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{Trans},
    ::Val{Trans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], B[j, l])
    end
    return acc
end

# Level 2 ----------------------------------------------------------------

function _gemv!(
    ::NativeBackend,
    trans::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    b::BigFloat,
    y::AbstractVector{BigFloat},
    p::Int,
)
    m, n = size(A)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    if trans === NoTrans
        @inbounds for i in 1:m
            MA.operate!(zero, acc)
            for l in 1:n
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], x[l])
            end
            _store_owned!(y[i], acc, buffer, a, b, alpha_is_one, beta_is_zero)
        end
    else
        @inbounds for j in 1:n
            MA.operate!(zero, acc)
            for l in 1:m
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, j], x[l])
            end
            _store_owned!(y[j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
        end
    end
    return y
end

@inline _trsm_coeff(A::AbstractMatrix{BigFloat}, row::Int, column::Int, transposed::Bool) =
    transposed ? A[column, row] : A[row, column]

function _trsv!(
    ::NativeBackend,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    A::AbstractMatrix{BigFloat},
    b::AbstractVector{BigFloat},
    p::Int,
)
    n = size(A, 1)
    n == 0 && return b
    effective_lower = (triangle === Lower) == (trans === NoTrans)
    transposed = trans === Trans
    unit_diagonal = diagonal === UnitDiagonal
    acc = _scratch(p)
    buffer = _scratch(p)
    difference = _scratch(p)
    if effective_lower
        @inbounds for i in 1:n
            MA.operate!(zero, acc)
            for k in 1:(i - 1)
                MA.buffered_operate!(buffer, MA.add_mul, acc, _trsm_coeff(A, i, k, transposed), b[k])
            end
            MA.operate_to!(difference, -, b[i], acc)
            if unit_diagonal
                MA.operate_to!(b[i], copy, difference)
            else
                _mpfr_div!(b[i], difference, A[i, i])
            end
        end
    else
        @inbounds for i in n:-1:1
            MA.operate!(zero, acc)
            for k in (i + 1):n
                MA.buffered_operate!(buffer, MA.add_mul, acc, _trsm_coeff(A, i, k, transposed), b[k])
            end
            MA.operate_to!(difference, -, b[i], acc)
            if unit_diagonal
                MA.operate_to!(b[i], copy, difference)
            else
                _mpfr_div!(b[i], difference, A[i, i])
            end
        end
    end
    return b
end

function _syr!(
    ::NativeBackend,
    triangle::Triangle,
    a::BigFloat,
    x::AbstractVector{BigFloat},
    A::AbstractMatrix{BigFloat},
    p::Int,
)
    n = length(x)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    if triangle === Lower
        @inbounds for j in 1:n
            for i in j:n
                MA.operate_to!(buffer, *, x[i], x[j])
                alpha_is_one || MA.operate!(*, buffer, a)
                MA.operate!(+, A[i, j], buffer)
            end
        end
    else
        @inbounds for j in 1:n
            for i in 1:j
                MA.operate_to!(buffer, *, x[i], x[j])
                alpha_is_one || MA.operate!(*, buffer, a)
                MA.operate!(+, A[i, j], buffer)
            end
        end
    end
    return A
end

# Level 3 ----------------------------------------------------------------

function _gemm!(
    ::NativeBackend,
    ::Val{TA},
    ::Val{TB},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
) where {TA,TB}
    m, n = size(C)
    k = TA === NoTrans ? size(A, 2) : size(A, 1)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for j in 1:n
        for i in 1:m
            _dot_row_col!(acc, buffer, A, B, i, j, Val{TA}(), Val{TB}(), k)
            _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
        end
    end
    return C
end

@inline function _syrk_dot!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{NoTrans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], A[j, l])
    end
    return acc
end

@inline function _syrk_dot!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{Trans},
    k::Int,
)
    MA.operate!(zero, acc)
    @inbounds for l in 1:k
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], A[l, j])
    end
    return acc
end

function _syrk!(
    ::NativeBackend,
    triangle::Triangle,
    ::Val{T},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
) where {T}
    n = size(C, 1)
    k = T === NoTrans ? size(A, 2) : size(A, 1)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    if triangle === Lower
        @inbounds for j in 1:n
            for i in j:n
                _syrk_dot!(acc, buffer, A, i, j, Val{T}(), k)
                _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
            end
        end
    else
        @inbounds for j in 1:n
            for i in 1:j
                _syrk_dot!(acc, buffer, A, i, j, Val{T}(), k)
                _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
            end
        end
    end
    return C
end

@inline function _trmm_coeff(
    A::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    transposed::Bool,
    unit_diagonal::Bool,
    one_value::BigFloat,
)
    unit_diagonal && row == column && return one_value
    return transposed ? A[column, row] : A[row, column]
end

function _trmm!(
    ::NativeBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
)
    n = size(A, 1)
    n == 0 && return B
    effective_lower = (triangle === Lower) == (trans === NoTrans)
    transposed = trans === Trans
    unit_diagonal = diagonal === UnitDiagonal
    one_value = BigFloat(1; precision = p)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    if side === LeftSide
        order = effective_lower ? (n:-1:1) : (1:n)
        @inbounds for j in axes(B, 2)
            for i in order
                krange = effective_lower ? (1:i) : (i:n)
                MA.operate!(zero, acc)
                for k in krange
                    MA.buffered_operate!(
                        buffer,
                        MA.add_mul,
                        acc,
                        _trmm_coeff(A, i, k, transposed, unit_diagonal, one_value),
                        B[k, j],
                    )
                end
                if alpha_is_one
                    MA.operate_to!(B[i, j], copy, acc)
                else
                    MA.operate_to!(B[i, j], *, a, acc)
                end
            end
        end
    else
        order = effective_lower ? (1:n) : (n:-1:1)
        @inbounds for i in axes(B, 1)
            for j in order
                prange = effective_lower ? (j:n) : (1:j)
                MA.operate!(zero, acc)
                for q in prange
                    MA.buffered_operate!(
                        buffer,
                        MA.add_mul,
                        acc,
                        B[i, q],
                        _trmm_coeff(A, q, j, transposed, unit_diagonal, one_value),
                    )
                end
                if alpha_is_one
                    MA.operate_to!(B[i, j], copy, acc)
                else
                    MA.operate_to!(B[i, j], *, a, acc)
                end
            end
        end
    end
    return B
end

function _trsm!(
    ::NativeBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
)
    n = size(A, 1)
    n == 0 && return B
    effective_lower = (triangle === Lower) == (trans === NoTrans)
    transposed = trans === Trans
    unit_diagonal = diagonal === UnitDiagonal
    acc = _scratch(p)
    buffer = _scratch(p)
    difference = _scratch(p)
    _native_scale!(B, a)
    if side === LeftSide
        if effective_lower
            @inbounds for j in axes(B, 2)
                for i in 1:n
                    MA.operate!(zero, acc)
                    for k in 1:(i - 1)
                        MA.buffered_operate!(buffer, MA.add_mul, acc, _trsm_coeff(A, i, k, transposed), B[k, j])
                    end
                    MA.operate_to!(difference, -, B[i, j], acc)
                    if unit_diagonal
                        MA.operate_to!(B[i, j], copy, difference)
                    else
                        _mpfr_div!(B[i, j], difference, A[i, i])
                    end
                end
            end
        else
            @inbounds for j in axes(B, 2)
                for i in n:-1:1
                    MA.operate!(zero, acc)
                    for k in (i + 1):n
                        MA.buffered_operate!(buffer, MA.add_mul, acc, _trsm_coeff(A, i, k, transposed), B[k, j])
                    end
                    MA.operate_to!(difference, -, B[i, j], acc)
                    if unit_diagonal
                        MA.operate_to!(B[i, j], copy, difference)
                    else
                        _mpfr_div!(B[i, j], difference, A[i, i])
                    end
                end
            end
        end
    else
        if effective_lower
            @inbounds for i in axes(B, 1)
                for j in n:-1:1
                    MA.operate!(zero, acc)
                    for k in (j + 1):n
                        MA.buffered_operate!(buffer, MA.add_mul, acc, B[i, k], _trsm_coeff(A, k, j, transposed))
                    end
                    MA.operate_to!(difference, -, B[i, j], acc)
                    if unit_diagonal
                        MA.operate_to!(B[i, j], copy, difference)
                    else
                        _mpfr_div!(B[i, j], difference, A[j, j])
                    end
                end
            end
        else
            @inbounds for i in axes(B, 1)
                for j in 1:n
                    MA.operate!(zero, acc)
                    for k in 1:(j - 1)
                        MA.buffered_operate!(buffer, MA.add_mul, acc, B[i, k], _trsm_coeff(A, k, j, transposed))
                    end
                    MA.operate_to!(difference, -, B[i, j], acc)
                    if unit_diagonal
                        MA.operate_to!(B[i, j], copy, difference)
                    else
                        _mpfr_div!(B[i, j], difference, A[j, j])
                    end
                end
            end
        end
    end
    return B
end

# Cholesky ---------------------------------------------------------------

function _cholesky!(::NativeBackend, A::AbstractMatrix{BigFloat}, triangle::Triangle, p::Int)
    triangle === Lower ||
        _unsupported(NativeBackend(), :cholesky, "NativeBackend supports triangle=Lower only in phase 1")
    k = size(A, 1)
    k == 0 && return 0
    acc = _scratch(p)
    buffer = _scratch(p)
    difference = _scratch(p)
    @inbounds for j in 1:k
        if j > 1
            MA.operate!(zero, acc)
            for l in 1:(j - 1)
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[j, l], A[j, l])
            end
            MA.operate_to!(difference, -, A[j, j], acc)
            djj = difference
        else
            djj = A[j, j]
        end
        djj <= 0 && return j
        _mpfr_sqrt!(A[j, j], djj)
        Ljj = A[j, j]
        for i in (j + 1):k
            if j > 1
                MA.operate!(zero, acc)
                for l in 1:(j - 1)
                    MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], A[j, l])
                end
                MA.operate_to!(difference, -, A[i, j], acc)
                numerator = difference
            else
                numerator = A[i, j]
            end
            _mpfr_div!(A[i, j], numerator, Ljj)
        end
    end
    return 0
end

function _cholesky_solve!(
    ::NativeBackend,
    L::AbstractMatrix{BigFloat},
    triangle::Triangle,
    p::Int,
    rhs::AbstractVecOrMat{BigFloat},
)
    triangle === Lower ||
        _unsupported(NativeBackend(), :factor_solve, "NativeBackend supports triangle=Lower only in phase 1")
    n = size(L, 1)
    n == 0 && return rhs
    acc = _scratch(p)
    buffer = _scratch(p)
    difference = _scratch(p)
    @inbounds for column in axes(rhs, 2)
        for row in 1:n
            MA.operate!(zero, acc)
            for k in 1:(row - 1)
                MA.buffered_operate!(buffer, MA.add_mul, acc, L[row, k], rhs[k, column])
            end
            MA.operate_to!(difference, -, rhs[row, column], acc)
            _mpfr_div!(rhs[row, column], difference, L[row, row])
        end
        for row in n:-1:1
            MA.operate!(zero, acc)
            for k in (row + 1):n
                MA.buffered_operate!(buffer, MA.add_mul, acc, L[k, row], rhs[k, column])
            end
            MA.operate_to!(difference, -, rhs[row, column], acc)
            _mpfr_div!(rhs[row, column], difference, L[row, row])
        end
    end
    return rhs
end
