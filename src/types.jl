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
    gemmt = true,
    symv = true,
    syr2k = true,
    trsm = true,
    trsv = true,
    trmm = true,
    cholesky = true,
    cholesky_triangles = (Lower,),
    ldlt = true,
    rank_revealing_qr = true,
    lu = true,
    multi_rhs = true,
    residual = true,
    backward_error = true,
    higher_precision_residual = true,
    refinement = true,
    precision_conversion = true,
    factor_solve = true,
    threading = true,
    ownership_safe = true,
)

capabilities(::GenericBackend) = (
    gemm = true,
    gemv = true,
    syrk = true,
    gemmt = true,
    symv = true,
    syr2k = true,
    trsm = true,
    trsv = true,
    trmm = true,
    cholesky = true,
    cholesky_triangles = (Lower, Upper),
    ldlt = true,
    rank_revealing_qr = true,
    lu = true,
    multi_rhs = true,
    residual = true,
    backward_error = true,
    higher_precision_residual = true,
    refinement = true,
    precision_conversion = true,
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
    FactorStatus

Machine-readable result of a factorization. `kind` is one of the symbols
`:success`, `:not_positive_definite`, `:nonfinite`, `:singular`, or
`:pivot_failure`; `position` is an optional 1-based pivot/failure position
(`nothing` when it does not apply). This replaces the previous ambiguous
integer sentinels (such as `-1` for a non-finite triangle) so that Cholesky,
LDLᵀ, QR, and LU factors share one stable, extensible protocol.
"""
struct FactorStatus
    kind::Symbol
    position::Union{Nothing,Int}
end

const SUCCESS_STATUS = FactorStatus(:success, nothing)

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
  * `status`: a [`FactorStatus`](@ref) describing the result (`check=false`
    only; `check=true` throws instead of returning a failed factor).
"""
struct BFLACholeskyFactor{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractBFLAFactor
    factors::M
    backend::B
    triangle::Triangle
    precision_bits::Int
    status::FactorStatus
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
    factor_status(F) -> FactorStatus

Machine-readable factorization result. On success `kind` is `:success`; a
non-positive-definite matrix yields `:not_positive_definite` with the 1-based
failure pivot in `position`; a non-finite authoritative triangle yields
`:nonfinite` with `position === nothing`.
"""
factor_status(F::BFLACholeskyFactor) = F.status

"""
    factor_kind(F) -> Symbol

Kind of factorization (`:cholesky`, `:ldlt`, `:qr`, `:lu`, ...).
"""
factor_kind(::BFLACholeskyFactor) = :cholesky

"""
    issuccess(F) -> Bool

Whether the factorization succeeded.
"""
issuccess(F::BFLACholeskyFactor) = F.status.kind === :success

Base.size(F::BFLACholeskyFactor) = size(F.factors)
Base.size(F::BFLACholeskyFactor, dimension::Integer) =
    size(F.factors, dimension)
Base.eltype(::BFLACholeskyFactor{M,B}) where {M,B} = BigFloat
