"""
    AbstractBFLABackend

Supertype for all BFLA execution backends. Every public kernel and
factorization accepts an explicit backend as its first positional argument so
that a caller can audit which code path produced a result.
"""
abstract type AbstractBFLABackend end

"""
    NativeBackend

The default, MPFR-native backend. Kernels are extracted from the SDPX legacy
BigFloat dense kernels and keep independent MPFR ownership throughout. This
backend never converts to `Float64` and never silently delegates to
[`GenericBackend`](@ref).
"""
struct NativeBackend <: AbstractBFLABackend end

"""
    GenericBackend

Reference backend built on Julia `LinearAlgebra` generic methods. It is used
for numerical cross-checking and as a performance baseline, not as an implicit
fallback for the Native backend.
"""
struct GenericBackend <: AbstractBFLABackend end

"""
    DEFAULT_BACKEND

Convenience singleton equal to [`NativeBackend`](@ref)().
"""
const DEFAULT_BACKEND = NativeBackend()

"""
    TransposeOp

Whether an operand is used as-is or transposed.

  * `NoTrans`: use `A` directly.
  * `Trans`: use `transpose(A)`.
"""
@enum TransposeOp NoTrans Trans

"""
    Triangle

Which triangle of a symmetric or triangular matrix is authoritative.

  * `Lower`: lower triangle (`i >= j`).
  * `Upper`: upper triangle (`i <= j`).
"""
@enum Triangle Lower Upper

"""
    Side

Whether a triangular matrix is applied on the left or the right.
"""
@enum Side LeftSide RightSide

"""
    DiagonalKind

Whether the diagonal of a triangular matrix is implicit unit or stored.

  * `UnitDiagonal`: the diagonal is assumed to be all ones.
  * `NonUnitDiagonal`: the diagonal is read from storage.
"""
@enum DiagonalKind UnitDiagonal NonUnitDiagonal

"""
    capabilities(backend) -> NamedTuple

Return a fixed, auditable description of the operations a backend supports.
This is a pure query; it performs no computation and is never used as an
implicit fallback gate. Fields are stable, machine-readable, and
backend-specific; `cholesky_triangles` enumerates the authoritative triangles
the backend can factor, because `NativeBackend` supports lower-triangular
Cholesky only while `GenericBackend` supports both lower and upper.
"""
function capabilities end

capabilities(::NativeBackend) = (
    gemm = true,
    gemv = true,
    syrk = true,
    trsm = true,
    trsv = true,
    trmm = true,
    cholesky = true,
    cholesky_triangles = (:lower,),
    factor_solve = true,
    threading = false,
    ownership_safe = true,
)

capabilities(::GenericBackend) = (
    gemm = true,
    gemv = true,
    syrk = true,
    trsm = true,
    trsv = true,
    trmm = true,
    cholesky = true,
    cholesky_triangles = (:lower, :upper),
    factor_solve = true,
    threading = false,
    ownership_safe = true,
)

"""
    PrecisionMismatch <: Exception

Thrown when a `BigFloat` operand does not carry a uniform MPFR precision. This
includes both intra-array mismatch (one element has a different precision than
the rest) and cross-operand mismatch. `index` is the 1-based linear index of the
offending element for intra-array failures, or `nothing` when the mismatch is
between whole operands.
"""
struct PrecisionMismatch <: Exception
    expected::Int
    found::Int
    index::Union{Nothing,Int}
end

function Base.showerror(io::IO, err::PrecisionMismatch)
    if err.index === nothing
        print(io, "PrecisionMismatch: expected BigFloat precision ",
              err.expected, " bits, found ", err.found, " bits")
    else
        print(io, "PrecisionMismatch: expected BigFloat precision ",
              err.expected, " bits, found ", err.found,
              " bits at linear index ", err.index)
    end
end

"""
    UnsupportedOperation

Thrown when a backend does not implement a requested operation. BFLA never
falls back to another backend at the operation level; an unsupported operation
is a hard, identifiable error.
"""
struct UnsupportedOperation <: Exception
    backend::AbstractBFLABackend
    operation::Symbol
    message::String
end

function Base.showerror(io::IO, err::UnsupportedOperation)
    print(io, "UnsupportedOperation: backend `", typeof(err.backend),
          "` does not support `", err.operation, "`: ", err.message)
end

@noinline function _unsupported(backend::AbstractBFLABackend, operation::Symbol, detail::AbstractString)
    throw(UnsupportedOperation(backend, operation, detail))
end

"""
    AbstractBFLAFactor

Supertype for factorizations owned by BFLA. A factor handle records the backend
that produced it; every solve dispatches through that backend rather than a
different one.
"""
abstract type AbstractBFLAFactor end

"""
    BFLACholeskyFactor{M,B} <: AbstractBFLAFactor

Lower- or upper-triangular Cholesky factor `L` (or `U`) of a symmetric
positive-definite `BigFloat` matrix.

Fields:

  * `factors`: the matrix holding the authoritative triangle in place.
  * `backend`: the backend that produced the factor.
  * `triangle`: which triangle of `factors` is authoritative.
  * `precision_bits`: the MPFR precision of the factor storage.
  * `info`: zero on success, or the 1-based pivot index where the
    factorization stopped (`check=false` only).
"""
struct BFLACholeskyFactor{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractBFLAFactor
    factors::M
    backend::B
    triangle::Triangle
    precision_bits::Int
    info::Int
end

"""
    factor_matrix(F) -> AbstractMatrix{BigFloat}

Return the matrix holding the factored data. Only the authoritative triangle is
meaningful.
"""
factor_matrix(F::BFLACholeskyFactor) = F.factors

"""
    factor_backend(F) -> AbstractBFLABackend

Return the backend that produced `F`.
"""
factor_backend(F::BFLACholeskyFactor) = F.backend

"""
    factor_triangle(F) -> Triangle

Return the authoritative triangle of `F`.
"""
factor_triangle(F::BFLACholeskyFactor) = F.triangle

"""
    factor_precision(F) -> Int

Return the MPFR precision, in bits, of the factor storage.
"""
factor_precision(F::BFLACholeskyFactor) = F.precision_bits

"""
    factor_status(F) -> Int

Zero for a successful factorization; otherwise the 1-based pivot index where
the factorization stopped.
"""
factor_status(F::BFLACholeskyFactor) = F.info

"""
    issuccess(F) -> Bool

Whether the factorization succeeded.
"""
issuccess(F::BFLACholeskyFactor) = iszero(F.info)

Base.size(F::BFLACholeskyFactor) = size(F.factors)
Base.eltype(::BFLACholeskyFactor{M,B}) where {M,B} = BigFloat
