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

# Kernel dispatch. These entry points receive an already-validated `p` and
# select blocked/threaded/unblocked Native kernels. In the initial release they
# route to the unblocked kernels; blocked and threaded paths are added in later
# phases and selected via `config`. GenericBackend ignores `config`
# because it is a single-threaded reference backend.

function _gemm_dispatch!(backend::NativeBackend, ::Val{TA}, ::Val{TB}, a, A, B, b, C, p, config) where {TA,TB}
    if config.gemm_block > 0
        return _gemm_blocked!(backend, Val{TA}(), Val{TB}(), a, A, B, b, C, p, config.gemm_block)
    end
    if config.thread_count > 1
        return _gemm_threaded!(backend, Val{TA}(), Val{TB}(), a, A, B, b, C, p, config)
    end
    return _gemm!(backend, Val{TA}(), Val{TB}(), a, A, B, b, C, p)
end

function _syrk_dispatch!(backend::NativeBackend, triangle, ::Val{T}, a, A, b, C, p, config) where {T}
    if config.syrk_block > 0
        return _syrk_blocked!(backend, triangle, Val{T}(), a, A, b, C, p, config.syrk_block)
    end
    if config.thread_count > 1
        return _syrk_threaded!(backend, triangle, Val{T}(), a, A, b, C, p, config)
    end
    return _syrk!(backend, triangle, Val{T}(), a, A, b, C, p)
end

function _trsm_dispatch!(backend::NativeBackend, side, triangle, trans, diagonal, a, A, B, p, config)
    if config.trsm_block > 0
        return _trsm_blocked!(backend, side, triangle, trans, diagonal, a, A, B, p, config.trsm_block)
    end
    if config.thread_count > 1
        return _trsm_threaded!(backend, side, triangle, trans, diagonal, a, A, B, p, config)
    end
    return _trsm!(backend, side, triangle, trans, diagonal, a, A, B, p)
end

function _cholesky_dispatch!(backend::NativeBackend, A, triangle, p, config)
    if config.cholesky_block > 0
        return _cholesky_blocked!(backend, A, triangle, p, config.cholesky_block)
    end
    return _cholesky!(backend, A, triangle, p)
end

# GenericBackend ignores config.
function _gemm_dispatch!(backend::GenericBackend, ::Val{TA}, ::Val{TB}, a, A, B, b, C, p, config) where {TA,TB}
    return _gemm!(backend, Val{TA}(), Val{TB}(), a, A, B, b, C, p)
end
function _syrk_dispatch!(backend::GenericBackend, triangle, ::Val{T}, a, A, b, C, p, config) where {T}
    return _syrk!(backend, triangle, Val{T}(), a, A, b, C, p)
end
function _trsm_dispatch!(backend::GenericBackend, side, triangle, trans, diagonal, a, A, B, p, config)
    return _trsm!(backend, side, triangle, trans, diagonal, a, A, B, p)
end
function _cholesky_dispatch!(backend::GenericBackend, A, triangle, p, config)
    return _cholesky!(backend, A, triangle, p)
end

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

@inline _ga(A::AbstractMatrix{BigFloat}, i::Int, l::Int, ::Val{NoTrans}) = A[i, l]
@inline _ga(A::AbstractMatrix{BigFloat}, i::Int, l::Int, ::Val{Trans}) = A[l, i]
@inline _gb(B::AbstractMatrix{BigFloat}, l::Int, j::Int, ::Val{NoTrans}) = B[l, j]
@inline _gb(B::AbstractMatrix{BigFloat}, l::Int, j::Int, ::Val{Trans}) = B[j, l]

@inline function _accum_row_col!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{TA},
    ::Val{TB},
    k0::Int,
    k1::Int,
) where {TA,TB}
    @inbounds for l in k0:k1
        MA.buffered_operate!(buffer, MA.add_mul, acc, _ga(A, i, l, Val{TA}()), _gb(B, l, j, Val{TB}()))
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

function _symv!(
    ::NativeBackend,
    triangle::Triangle,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    b::BigFloat,
    y::AbstractVector{BigFloat},
    p::Int,
)
    n = length(x)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for i in 1:n
        MA.operate!(zero, acc)
        if triangle === Lower
            for j in 1:i
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, j], x[j])
            end
            for j in (i + 1):n
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[j, i], x[j])
            end
        else
            for j in 1:(i - 1)
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[j, i], x[j])
            end
            for j in i:n
                MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, j], x[j])
            end
        end
        _store_owned!(y[i], acc, buffer, a, b, alpha_is_one, beta_is_zero)
    end
    return y
end

# Level 3 ----------------------------------------------------------------

function _gemm!(
    ::NativeBackend,
    ::Val{NoTrans},
    ::Val{NoTrans},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for j in axes(C, 2)
        B_column = view(B, :, j)
        for i in axes(C, 1)
            A_row = view(A, i, :)
            MA.operate!(zero, acc)
            for index in eachindex(A_row, B_column)
                MA.buffered_operate!(
                    buffer,
                    MA.add_mul,
                    acc,
                    A_row[index],
                    B_column[index],
                )
            end
            _store_owned!(
                C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero,
            )
        end
    end
    return C
end

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

function _gemmt!(
    ::NativeBackend,
    triangle::Triangle,
    ::Val{TA},
    ::Val{TB},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
) where {TA,TB}
    n = size(C, 1)
    k = TA === NoTrans ? size(A, 2) : size(A, 1)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for j in 1:n
        ilo, ihi = triangle === Lower ? (j, n) : (1, j)
        for i in ilo:ihi
            MA.operate!(zero, acc)
            for l in 1:k
                MA.buffered_operate!(buffer, MA.add_mul, acc, _ga(A, i, l, Val{TA}()), _gb(B, l, j, Val{TB}()))
            end
            _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
        end
    end
    return C
end

function _syr2k!(
    ::NativeBackend,
    triangle::Triangle,
    ::Val{T},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
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
    @inbounds for j in 1:n
        ilo, ihi = triangle === Lower ? (j, n) : (1, j)
        for i in ilo:ihi
            MA.operate!(zero, acc)
            for l in 1:k
                if T === NoTrans
                    # A[i,l]*B[j,l] + B[i,l]*A[j,l]
                    MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], B[j, l])
                    MA.buffered_operate!(buffer, MA.add_mul, acc, B[i, l], A[j, l])
                else
                    # A[l,i]*B[l,j] + B[l,i]*A[l,j]
                    MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], B[l, j])
                    MA.buffered_operate!(buffer, MA.add_mul, acc, B[l, i], A[l, j])
                end
            end
            _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
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

# Blocked single-threaded kernels. Selected explicitly via `KernelConfig`.
#
# `_gemm_blocked!` and `_syrk_blocked!` partition the reduction dimension into
# tiles and keep the per-entry multiply-accumulate sequence in ascending
# `l` order, so they are bit-identical to the unblocked kernels. `_cholesky_blocked!`
# and `_trsm_blocked!` use the standard right-looking / block-triangular
# algorithms and are validated by residual rather than bit-parity.

function _gemm_blocked!(
    ::NativeBackend,
    ::Val{TA},
    ::Val{TB},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
    bs::Int,
) where {TA,TB}
    m, n = size(C)
    k = TA === NoTrans ? size(A, 2) : size(A, 1)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for j0 in 1:bs:n
        j1 = min(j0 + bs - 1, n)
        for i0 in 1:bs:m
            i1 = min(i0 + bs - 1, m)
            for j in j0:j1
                for i in i0:i1
                    MA.operate!(zero, acc)
                    for k0 in 1:bs:k
                        _accum_row_col!(acc, buffer, A, B, i, j, Val{TA}(), Val{TB}(), k0, min(k0 + bs - 1, k))
                    end
                    _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
                end
            end
        end
    end
    return C
end

function _syrk_blocked!(
    ::NativeBackend,
    triangle::Triangle,
    ::Val{T},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
    bs::Int,
) where {T}
    n = size(C, 1)
    k = T === NoTrans ? size(A, 2) : size(A, 1)
    acc = _scratch(p)
    buffer = _scratch(p)
    alpha_is_one = isone(a)
    beta_is_zero = iszero(b)
    @inbounds for j in 1:n
        ilo, ihi = triangle === Lower ? (j, n) : (1, j)
        for i in ilo:ihi
            MA.operate!(zero, acc)
            for k0 in 1:bs:k
                _accum_syrk!(acc, buffer, A, i, j, Val{T}(), k0, min(k0 + bs - 1, k))
            end
            _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
        end
    end
    return C
end

@inline function _accum_syrk!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{NoTrans},
    k0::Int,
    k1::Int,
)
    @inbounds for l in k0:k1
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], A[j, l])
    end
    return acc
end

@inline function _accum_syrk!(
    acc::BigFloat,
    buffer::BigFloat,
    A::AbstractMatrix{BigFloat},
    i::Int,
    j::Int,
    ::Val{Trans},
    k0::Int,
    k1::Int,
)
    @inbounds for l in k0:k1
        MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], A[l, j])
    end
    return acc
end

function _cholesky_blocked!(::NativeBackend, A::AbstractMatrix{BigFloat}, triangle::Triangle, p::Int, bs::Int)
    triangle === Lower ||
        _unsupported(NativeBackend(), :cholesky, "NativeBackend supports triangle=Lower only in phase 1")
    n = size(A, 1)
    n == 0 && return 0
    one = BigFloat(1; precision = p)
    negone = BigFloat(-1; precision = p)
    j0 = 1
    while j0 <= n
        j1 = min(j0 + bs - 1, n)
        A11 = view(A, j0:j1, j0:j1)
        info = _cholesky!(NativeBackend(), A11, Lower, p)
        info != 0 && return j0 + info - 1
        if j1 < n
            A21 = view(A, (j1 + 1):n, j0:j1)
            # L21 = A21 * inv(L11)^T
            _trsm!(NativeBackend(), RightSide, Lower, Trans, NonUnitDiagonal, one, A11, A21, p)
            A22 = view(A, (j1 + 1):n, (j1 + 1):n)
            # A22 -= L21 * L21^T
            _syrk!(NativeBackend(), Lower, Val(NoTrans), negone, A21, one, A22, p)
        end
        j0 = j1 + 1
    end
    return 0
end

# Explicit multithreading. Each task owns a disjoint output region and allocates
# its own MPFR scratch, so no mutable accumulator is shared across threads.
# Reduction order within one output entry is unchanged, so threaded results are
# bit-identical to serial results.

function _gemm_threaded!(
    ::NativeBackend,
    ::Val{TA},
    ::Val{TB},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
    config::KernelConfig,
) where {TA,TB}
    m, n = size(C)
    k = TA === NoTrans ? size(A, 2) : size(A, 1)
    workers = _worker_count(config, n)
    if workers <= 1
        return _gemm!(NativeBackend(), Val{TA}(), Val{TB}(), a, A, B, b, C, p)
    end
    chunk = cld(n, workers)
    @sync for w in 1:workers
        j0 = (w - 1) * chunk + 1
        j0 > n && continue
        j1 = min(w * chunk, n)
        Threads.@spawn begin
            acc = _scratch(p)
            buffer = _scratch(p)
            alpha_is_one = isone(a)
            beta_is_zero = iszero(b)
            @inbounds for j in j0:j1
                for i in 1:m
                    MA.operate!(zero, acc)
                    for l in 1:k
                        MA.buffered_operate!(buffer, MA.add_mul, acc, _ga(A, i, l, Val{TA}()), _gb(B, l, j, Val{TB}()))
                    end
                    _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
                end
            end
        end
    end
    return C
end

function _syrk_threaded!(
    ::NativeBackend,
    triangle::Triangle,
    ::Val{T},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
    config::KernelConfig,
) where {T}
    n = size(C, 1)
    k = T === NoTrans ? size(A, 2) : size(A, 1)
    workers = _worker_count(config, n)
    if workers <= 1
        return _syrk!(NativeBackend(), triangle, Val{T}(), a, A, b, C, p)
    end
    chunk = cld(n, workers)
    @sync for w in 1:workers
        j0 = (w - 1) * chunk + 1
        j0 > n && continue
        j1 = min(w * chunk, n)
        Threads.@spawn begin
            acc = _scratch(p)
            buffer = _scratch(p)
            alpha_is_one = isone(a)
            beta_is_zero = iszero(b)
            @inbounds for j in j0:j1
                ilo, ihi = triangle === Lower ? (j, n) : (1, j)
                for i in ilo:ihi
                    MA.operate!(zero, acc)
                    for l in 1:k
                        if T === NoTrans
                            MA.buffered_operate!(buffer, MA.add_mul, acc, A[i, l], A[j, l])
                        else
                            MA.buffered_operate!(buffer, MA.add_mul, acc, A[l, i], A[l, j])
                        end
                    end
                    _store_owned!(C[i, j], acc, buffer, a, b, alpha_is_one, beta_is_zero)
                end
            end
        end
    end
    return C
end

function _trsm_threaded!(
    ::NativeBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
    config::KernelConfig,
)
    rhs_columns = side === LeftSide ? size(B, 2) : size(B, 1)
    workers = _worker_count(config, rhs_columns)
    if workers <= 1
        return _trsm!(NativeBackend(), side, triangle, trans, diagonal, a, A, B, p)
    end
    _native_scale!(B, a)
    one = BigFloat(1; precision = p)
    if side === LeftSide
        chunk = cld(size(B, 2), workers)
        @sync for w in 1:workers
            c0 = (w - 1) * chunk + 1
            c0 > size(B, 2) && continue
            c1 = min(w * chunk, size(B, 2))
            Threads.@spawn begin
                Bcol = view(B, :, c0:c1)
                _trsm!(NativeBackend(), LeftSide, triangle, trans, diagonal, one, A, Bcol, p)
            end
        end
    else
        chunk = cld(size(B, 1), workers)
        @sync for w in 1:workers
            r0 = (w - 1) * chunk + 1
            r0 > size(B, 1) && continue
            r1 = min(w * chunk, size(B, 1))
            Threads.@spawn begin
                Brow = view(B, r0:r1, :)
                _trsm!(NativeBackend(), RightSide, triangle, trans, diagonal, one, A, Brow, p)
            end
        end
    end
    return B
end

function _trsm_blocked!(
    ::NativeBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
    bs::Int,
)
    n = size(A, 1)
    n == 0 && return B
    _native_scale!(B, a)
    one = BigFloat(1; precision = p)
    negone = BigFloat(-1; precision = p)
    # Left-side blocked forward/back substitution. Right-side falls back to the
    # unblocked kernel (right-side blocking is not on the current hot path).
    if side === LeftSide && trans === NoTrans
        if triangle === Lower
            j0 = 1
            while j0 <= n
                j1 = min(j0 + bs - 1, n)
                L11 = view(A, j0:j1, j0:j1)
                B1 = view(B, j0:j1, :)
                _trsm!(NativeBackend(), LeftSide, Lower, NoTrans, diagonal, one, L11, B1, p)
                if j1 < n
                    L21 = view(A, (j1 + 1):n, j0:j1)
                    B2 = view(B, (j1 + 1):n, :)
                    _gemm!(NativeBackend(), Val(NoTrans), Val(NoTrans), negone, L21, B1, one, B2, p)
                end
                j0 = j1 + 1
            end
        else
            j0 = n
            while j0 >= 1
                j1 = max(j0 - bs + 1, 1)
                U11 = view(A, j1:j0, j1:j0)
                B1 = view(B, j1:j0, :)
                _trsm!(NativeBackend(), LeftSide, Upper, NoTrans, diagonal, one, U11, B1, p)
                if j1 > 1
                    U01 = view(A, 1:(j1 - 1), j1:j0)
                    B0 = view(B, 1:(j1 - 1), :)
                    _gemm!(NativeBackend(), Val(NoTrans), Val(NoTrans), negone, U01, B1, one, B0, p)
                end
                j0 = j1 - 1
            end
        end
    else
        return _trsm!(NativeBackend(), side, triangle, trans, diagonal, one, A, B, p)
    end
    return B
end

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
            L_j = view(A, j, 1:(j - 1))
            MA.operate!(zero, acc)
            for index in eachindex(L_j)
                MA.buffered_operate!(
                    buffer, MA.add_mul, acc, L_j[index], L_j[index],
                )
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
                L_i = view(A, i, 1:(j - 1))
                L_j = view(A, j, 1:(j - 1))
                MA.operate!(zero, acc)
                for index in eachindex(L_i, L_j)
                    MA.buffered_operate!(
                        buffer,
                        MA.add_mul,
                        acc,
                        L_i[index],
                        L_j[index],
                    )
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
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
)
    triangle === Lower ||
        _unsupported(NativeBackend(), :factor_solve, "NativeBackend supports triangle=Lower only in phase 1")
    n = size(L, 1)
    n == 0 && return rhs
    acc = _solve_scratch(workspace, workspace_worker, 1, p)
    buffer = _solve_scratch(workspace, workspace_worker, 2, p)
    difference = _solve_scratch(workspace, workspace_worker, 3, p)
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
