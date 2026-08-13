# GenericBackend reference kernels. They delegate to Julia `LinearAlgebra`
# generic methods (which are correct for BigFloat but allocate freely) and run
# inside a scoped `setprecision` so intermediate values carry the target
# precision rather than the ambient global precision.

# A reference backend may serialize the precision context. `NativeBackend` does
# not use this lock and remains fully explicit and lock-free.
const _GENERIC_PRECISION_LOCK = ReentrantLock()

function _with_precision(f::Function, p::Int)
    lock(_GENERIC_PRECISION_LOCK)
    try
        return setprecision(BigFloat, p) do
            f()
        end
    finally
        unlock(_GENERIC_PRECISION_LOCK)
    end
end

function _generic_scale!(B::AbstractArray{BigFloat}, a::BigFloat)
    isone(a) && return B
    @inbounds for index in eachindex(B)
        B[index] = a * B[index]
    end
    return B
end

function _generic_triangular(
    A::AbstractMatrix{BigFloat},
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
)
    if triangle === Lower
        T = diagonal === UnitDiagonal ? UnitLowerTriangular(A) : LowerTriangular(A)
    else
        T = diagonal === UnitDiagonal ? UnitUpperTriangular(A) : UpperTriangular(A)
    end
    return trans === NoTrans ? T : transpose(T)
end

function _scal!(::GenericBackend, a::BigFloat, x::AbstractVector{BigFloat}, p::Int)
    _with_precision(p) do
        @inbounds for index in eachindex(x)
            x[index] = a * x[index]
        end
    end
    return x
end

function _axpy!(::GenericBackend, a::BigFloat, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat}, p::Int)
    _with_precision(p) do
        @inbounds for index in eachindex(x, y)
            y[index] = a * x[index] + y[index]
        end
    end
    return y
end

function _axpby!(::GenericBackend, a::BigFloat, x::AbstractVector{BigFloat}, b::BigFloat, y::AbstractVector{BigFloat}, p::Int)
    _with_precision(p) do
        @inbounds for index in eachindex(x, y)
            y[index] = a * x[index] + b * y[index]
        end
    end
    return y
end

function _dot(::GenericBackend, x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat}, p::Int)
    return _with_precision(p) do
        acc = zero(BigFloat)
        @inbounds for index in eachindex(x, y)
            acc += x[index] * y[index]
        end
        acc
    end
end

function _norminf(::GenericBackend, x::AbstractArray{BigFloat}, p::Int)
    return _with_precision(p) do
        m = BigFloat(0; precision = p)
        has_nan = false
        @inbounds for value in x
            if isnan(value)
                has_nan = true
                break
            end
            av = abs(value)
            av > m && (m = av)
        end
        has_nan ? BigFloat(NaN; precision = p) : m
    end
end

function _gemv!(
    ::GenericBackend,
    trans::TransposeOp,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    b::BigFloat,
    y::AbstractVector{BigFloat},
    p::Int,
)
    _with_precision(p) do
        opA = trans === NoTrans ? A : transpose(A)
        LinearAlgebra.mul!(y, opA, x, a, b)
    end
    return y
end

function _trsv!(
    ::GenericBackend,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    A::AbstractMatrix{BigFloat},
    b::AbstractVector{BigFloat},
    p::Int,
)
    _with_precision(p) do
        T = _generic_triangular(A, triangle, trans, diagonal)
        LinearAlgebra.ldiv!(T, b)
    end
    return b
end

function _syr!(
    ::GenericBackend,
    triangle::Triangle,
    a::BigFloat,
    x::AbstractVector{BigFloat},
    A::AbstractMatrix{BigFloat},
    p::Int,
)
    n = length(x)
    _with_precision(p) do
        @inbounds for j in 1:n
            if triangle === Lower
                for i in j:n
                    A[i, j] = a * x[i] * x[j] + A[i, j]
                end
            else
                for i in 1:j
                    A[i, j] = a * x[i] * x[j] + A[i, j]
                end
            end
        end
    end
    return A
end

function _symv!(
    ::GenericBackend,
    triangle::Triangle,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    b::BigFloat,
    y::AbstractVector{BigFloat},
    p::Int,
)
    n = length(x)
    _with_precision(p) do
        @inbounds for i in 1:n
            acc = zero(BigFloat)
            if triangle === Lower
                for j in 1:i
                    acc += A[i, j] * x[j]
                end
                for j in (i + 1):n
                    acc += A[j, i] * x[j]
                end
            else
                for j in 1:(i - 1)
                    acc += A[j, i] * x[j]
                end
                for j in i:n
                    acc += A[i, j] * x[j]
                end
            end
            y[i] = a * acc + b * y[i]
        end
    end
    return y
end

function _gemm!(
    ::GenericBackend,
    ::Val{TA},
    ::Val{TB},
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    b::BigFloat,
    C::AbstractMatrix{BigFloat},
    p::Int,
) where {TA,TB}
    _with_precision(p) do
        opA = TA === NoTrans ? A : transpose(A)
        opB = TB === NoTrans ? B : transpose(B)
        LinearAlgebra.mul!(C, opA, opB, a, b)
    end
    return C
end

function _syrk!(
    ::GenericBackend,
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
    _with_precision(p) do
        @inbounds for j in 1:n
            if triangle === Lower
                for i in j:n
                    acc = zero(BigFloat)
                    for l in 1:k
                        ai = T === NoTrans ? A[i, l] : A[l, i]
                        aj = T === NoTrans ? A[j, l] : A[l, j]
                        acc += ai * aj
                    end
                    C[i, j] = a * acc + b * C[i, j]
                end
            else
                for i in 1:j
                    acc = zero(BigFloat)
                    for l in 1:k
                        ai = T === NoTrans ? A[i, l] : A[l, i]
                        aj = T === NoTrans ? A[j, l] : A[l, j]
                        acc += ai * aj
                    end
                    C[i, j] = a * acc + b * C[i, j]
                end
            end
        end
    end
    return C
end

function _gemmt!(
    ::GenericBackend,
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
    _with_precision(p) do
        @inbounds for j in 1:n
            ilo, ihi = triangle === Lower ? (j, n) : (1, j)
            for i in ilo:ihi
                acc = zero(BigFloat)
                for l in 1:k
                    ai = TA === NoTrans ? A[i, l] : A[l, i]
                    bl = TB === NoTrans ? B[l, j] : B[j, l]
                    acc += ai * bl
                end
                C[i, j] = a * acc + b * C[i, j]
            end
        end
    end
    return C
end

function _syr2k!(
    ::GenericBackend,
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
    _with_precision(p) do
        @inbounds for j in 1:n
            ilo, ihi = triangle === Lower ? (j, n) : (1, j)
            for i in ilo:ihi
                acc = zero(BigFloat)
                for l in 1:k
                    if T === NoTrans
                        acc += A[i, l] * B[j, l] + B[i, l] * A[j, l]
                    else
                        acc += A[l, i] * B[l, j] + B[l, i] * A[l, j]
                    end
                end
                C[i, j] = a * acc + b * C[i, j]
            end
        end
    end
    return C
end

function _trmm!(
    ::GenericBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
)
    _with_precision(p) do
        T = _generic_triangular(A, triangle, trans, diagonal)
        _generic_scale!(B, a)
        if side === LeftSide
            LinearAlgebra.lmul!(T, B)
        else
            LinearAlgebra.rmul!(B, T)
        end
    end
    return B
end

function _trsm!(
    ::GenericBackend,
    side::Side,
    triangle::Triangle,
    trans::TransposeOp,
    diagonal::DiagonalKind,
    a::BigFloat,
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    p::Int,
)
    _with_precision(p) do
        T = _generic_triangular(A, triangle, trans, diagonal)
        _generic_scale!(B, a)
        if side === LeftSide
            LinearAlgebra.ldiv!(T, B)
        else
            LinearAlgebra.rdiv!(B, T)
        end
    end
    return B
end

function _cholesky!(::GenericBackend, A::AbstractMatrix{BigFloat}, triangle::Triangle, p::Int)
    return _with_precision(p) do
        uplo = triangle === Lower ? :L : :U
        F = LinearAlgebra.cholesky!(Symmetric(A, uplo); check=false)
        F.info
    end
end

function _cholesky_solve!(
    ::GenericBackend,
    L::AbstractMatrix{BigFloat},
    triangle::Triangle,
    p::Int,
    rhs::AbstractVecOrMat{BigFloat},
)
    return _with_precision(p) do
        T = triangle === Lower ? LowerTriangular(L) : UpperTriangular(L)
        if triangle === Lower
            LinearAlgebra.ldiv!(T, rhs)
            LinearAlgebra.ldiv!(transpose(T), rhs)
        else
            LinearAlgebra.ldiv!(transpose(T), rhs)
            LinearAlgebra.ldiv!(T, rhs)
        end
    end
    return rhs
end
