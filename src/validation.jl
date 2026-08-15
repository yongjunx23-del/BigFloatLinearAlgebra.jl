# Shared argument validation. All dimension, flag, alias, and precision checks
# happen once before entering a kernel; the inner loops do not re-validate.

"""
    _precision_bits(x) -> Union{Nothing,Int}

Return the MPFR precision, in bits, of a `BigFloat` scalar, or the uniform
precision of a non-empty `BigFloat` array. For arrays this performs a full scan
and throws [`PrecisionMismatch`](@ref) if any element differs from the first,
so a mixed-precision array fails closed at the public boundary. `nothing` is
returned for empty arrays and non-`BigFloat` values, which the checker ignores.
"""
_precision_bits(x::BigFloat) = precision(x)
_precision_bits(::Any) = nothing

function _precision_bits(A::AbstractArray{BigFloat})
    isempty(A) && return nothing
    p = precision(first(A))
    position = 0
    @inbounds for index in eachindex(A)
        position += 1
        q = precision(A[index])
        q == p || throw(PrecisionMismatch(p, q, position))
    end
    return p
end

"""
    _check_precision(args...) -> Union{Nothing,Int}

Verify that every `BigFloat` scalar and every non-empty `BigFloat` array in
`args` carries a single, uniform MPFR precision. Intra-array and cross-operand
mismatches fail closed with [`PrecisionMismatch`](@ref). Returns the common
precision, or `nothing` when no `BigFloat` value is present.

This runs exactly once at the public API boundary; the internal `_*` kernels
receive the already-validated precision and do not re-scan their inputs.
"""
function _check_precision(args...)
    common = nothing
    for arg in args
        p = _precision_bits(arg)
        p === nothing && continue
        if common === nothing
            common = p
        elseif common != p
            throw(PrecisionMismatch(common, p, nothing))
        end
    end
    return common
end

@noinline _require_precision(p::Nothing, op::AbstractString) =
    throw(ArgumentError("$op: could not determine a BigFloat precision"))

@inline _require_precision(p::Int, ::AbstractString) = p

_factor_precision_operands(::AbstractBFLAFactor) = ()

# Re-check mutable factor storage at every use boundary. Factor-specific
# finite/authority rules remain in each solve implementation.
function _validate_factor_precision(
    F::AbstractBFLAFactor,
    operation::AbstractString,
    operands...,
)
    actual = _require_precision(
        _check_precision(
            factor_matrix(F), _factor_precision_operands(F)..., operands...,
        ),
        operation,
    )
    recorded = factor_precision(F)
    actual == recorded || throw(PrecisionMismatch(recorded, actual, nothing))
    return actual
end

# Trusted factor use deliberately skips mutable factor storage. The caller has
# explicitly guaranteed that the factor and its metadata have not changed
# since a fully checked construction/use boundary. RHS storage is still
# scanned because it remains caller-owned mutable input for every solve.
function _validate_trusted_rhs_precision(
    F::AbstractBFLAFactor,
    operation::AbstractString,
    rhs::AbstractVecOrMat{BigFloat},
)
    recorded = factor_precision(F)
    isempty(rhs) && return recorded
    actual = _require_precision(_check_precision(rhs), operation)
    actual == recorded || throw(PrecisionMismatch(recorded, actual, nothing))
    return actual
end

"""
    _require_square(A, op)

Throw a `DimensionMismatch` unless `A` is square.
"""
function _require_square(A::AbstractMatrix, op::AbstractString)
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("$op: expected a square matrix"))
    return nothing
end

"""
    _require_no_alias(dest, src, op)

Throw an `ArgumentError` when `dest` may share storage with `src`.
"""
function _require_no_alias(dest, src, op::AbstractString)
    Base.mightalias(dest, src) &&
        throw(ArgumentError("$op: destination must not alias the source operand"))
    return nothing
end

"""
    _require_valid_transpose(op, name) -> TransposeOp
"""
function _require_valid_transpose(op::TransposeOp, name::AbstractString)
    (op === NoTrans || op === Trans) ||
        throw(ArgumentError("$name: invalid TransposeOp"))
    return op
end

function _require_valid_triangle(tri::Triangle, name::AbstractString)
    (tri === Lower || tri === Upper) ||
        throw(ArgumentError("$name: invalid Triangle"))
    return tri
end

function _require_valid_side(side::Side, name::AbstractString)
    (side === LeftSide || side === RightSide) ||
        throw(ArgumentError("$name: invalid Side"))
    return side
end

function _require_valid_diagonal(diag::DiagonalKind, name::AbstractString)
    (diag === UnitDiagonal || diag === NonUnitDiagonal) ||
        throw(ArgumentError("$name: invalid DiagonalKind"))
    return diag
end

"""
    _all_finite(A) -> Bool
"""
@inline function _all_finite(A::AbstractArray)
    @inbounds for index in eachindex(A)
        isfinite(A[index]) || return false
    end
    return true
end

"""
    _triangle_finite(A, triangle) -> Bool

Whether every entry in the authoritative triangle is finite.
"""
@inline function _triangle_finite(A::AbstractMatrix, triangle::Triangle)
    rows = axes(A, 1)
    columns = axes(A, 2)
    @inbounds for column in columns
        for row in rows
            if triangle === Lower
                row >= column || continue
            else
                row <= column || continue
            end
            isfinite(A[row, column]) || return false
        end
    end
    return true
end
