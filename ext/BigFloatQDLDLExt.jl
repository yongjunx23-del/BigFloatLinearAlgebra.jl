module BigFloatQDLDLExt

using BigFloatLinearAlgebra
using QDLDL
using SparseArrays
using LinearAlgebra: AbstractVecOrMat, istriu
using Base.Threads

const BFLA = BigFloatLinearAlgebra

struct QDLDLBackend <: BFLA.AbstractBFLABackend end

@inline function _mix(signature::UInt64, value::Integer)
    return (signature ⊻ reinterpret(UInt64, Int64(value))) * UInt64(0x100000001b3)
end

function _pattern_signature(A::SparseMatrixCSC, dsigns::AbstractVector{<:Integer})
    signature = UInt64(0xcbf29ce484222325)
    signature = _mix(signature, size(A, 1))
    signature = _mix(signature, size(A, 2))
    for value in A.colptr
        signature = _mix(signature, value)
    end
    for value in A.rowval
        signature = _mix(signature, value)
    end
    for value in dsigns
        signature = _mix(signature, value)
    end
    return signature
end

mutable struct BFLASparseLDLCache{Ti<:Integer} <: BFLA.AbstractFactorCache
    backend::QDLDLBackend
    matrix::SparseMatrixCSC{BigFloat,Ti}
    factor::Union{Nothing,QDLDL.QDLDLFactorisation{BigFloat,Ti}}
    indices::Vector{Ti}
    dsigns::Vector{Ti}
    pattern_signature::UInt64
    nrhs_capacity::Int
    n::Int
    precision_bits::Int
    status::BFLA.FactorStatus
    prepared::Bool
    symbolic_count::Int
    numeric_factor_count::Int
    solve_count::Int
end

function _ambient_guard(precision_bits::Int, operation::AbstractString)
    Threads.nthreads() == 1 || throw(ArgumentError(
        "$operation: QDLDL BigFloat prototype requires one Julia thread",
    ))
    ambient = precision(BigFloat)
    ambient == precision_bits || throw(BFLA.PrecisionMismatch(
        precision_bits, ambient, nothing,
    ))
    return nothing
end

function _validate_values(
    values::AbstractArray{BigFloat}, precision_bits::Int,
    operation::AbstractString,
)
    @inbounds for (index, value) in pairs(values)
        precision(value) == precision_bits || throw(BFLA.PrecisionMismatch(
            precision_bits, precision(value), Int(index),
        ))
        isfinite(value) || throw(DomainError(
            value, "$operation: non-finite sparse value at index $index",
        ))
    end
    return nothing
end

function _validate_pattern(
    A::SparseMatrixCSC{BigFloat,Ti}, dsigns::AbstractVector{<:Integer},
    precision_bits::Int,
) where {Ti<:Integer}
    n, m = size(A)
    n == m || throw(DimensionMismatch("QDLDL sparse LDL requires a square matrix"))
    istriu(A) || throw(ArgumentError(
        "QDLDL sparse LDL pattern must store the upper triangle",
    ))
    length(dsigns) == n || throw(DimensionMismatch(
        "QDLDL D-sign vector length must equal the matrix order",
    ))
    all(sign -> sign == -1 || sign == 1, dsigns) || throw(ArgumentError(
        "QDLDL D signs must be exactly +1 or -1",
    ))
    _validate_values(A.nzval, precision_bits, "sparse_ldlt_cache")
    @inbounds for column in 1:n
        A.colptr[column] < A.colptr[column + 1] || throw(ArgumentError(
            "QDLDL sparse LDL requires every structural column to be nonempty",
        ))
    end
    return nothing
end

function _factor_precision_ok(factor, precision_bits::Int)
    # QDLDL 0.4 publicly exposes the returned L and Dinv factors, but no
    # workspace-precision accessor. Inspect only those public factor outputs;
    # ambient/input guards protect the internal generic arithmetic.
    for storage in (factor.L.nzval, factor.Dinv.diag)
        @inbounds for value in storage
            precision(value) == precision_bits || return false
            isfinite(value) || return false
        end
    end
    return true
end

function BFLASparseLDLCache(
    pattern::SparseMatrixCSC{BigFloat,Ti}; precision_bits::Int,
    dsigns::AbstractVector{<:Integer}, nrhs::Integer=1,
    regularize_eps::Union{Nothing,BigFloat}=nothing,
    regularize_delta::Union{Nothing,BigFloat}=nothing,
) where {Ti<:Integer}
    _ambient_guard(precision_bits, "sparse_ldlt_cache")
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    _validate_pattern(pattern, dsigns, precision_bits)
    eps_value = regularize_eps === nothing ?
        BigFloat("1e-60"; precision=precision_bits) : regularize_eps
    delta_value = regularize_delta === nothing ?
        BigFloat("1e-40"; precision=precision_bits) : regularize_delta
    precision(eps_value) == precision_bits || throw(BFLA.PrecisionMismatch(
        precision_bits, precision(eps_value), nothing,
    ))
    precision(delta_value) == precision_bits || throw(BFLA.PrecisionMismatch(
        precision_bits, precision(delta_value), nothing,
    ))
    values = BFLA.owned_copy(pattern.nzval; precision_bits=precision_bits)
    matrix = SparseMatrixCSC(
        size(pattern, 1), size(pattern, 2), copy(pattern.colptr),
        copy(pattern.rowval), values,
    )
    signs = Ti[sign for sign in dsigns]
    indices = Ti.(eachindex(matrix.nzval))
    factor = QDLDL.qdldl(
        matrix; logical=true, Dsigns=signs,
        regularize_eps=eps_value, regularize_delta=delta_value,
    )
    _factor_precision_ok(factor, precision_bits) || throw(ArgumentError(
        "QDLDL symbolic workspace does not preserve BigFloat precision",
    ))
    n = size(matrix, 1)
    return BFLASparseLDLCache{Ti}(
        QDLDLBackend(), matrix, factor, indices, signs,
        _pattern_signature(matrix, signs), Int(nrhs), n, precision_bits,
        BFLA.FactorStatus(:unprepared, nothing), true, 1, 0, 0,
    )
end

BFLA.factor_kind(::BFLASparseLDLCache) = :sparse_ldlt
BFLA.factor_matrix(cache::BFLASparseLDLCache) = cache.matrix

function _validate_numeric(cache::BFLASparseLDLCache{Ti}, A) where {Ti}
    A isa SparseMatrixCSC{BigFloat,Ti} || throw(ArgumentError(
        "QDLDL sparse LDL numeric matrix must preserve BigFloat/index types",
    ))
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "QDLDL sparse LDL numeric dimensions differ from the frozen pattern",
    ))
    A.colptr == cache.matrix.colptr && A.rowval == cache.matrix.rowval ||
        throw(ArgumentError("QDLDL sparse LDL pattern drift"))
    _validate_values(A.nzval, cache.precision_bits, "factorize!")
    return nothing
end

function BFLA.factorize!(
    cache::BFLASparseLDLCache{Ti}, A::SparseMatrixCSC{BigFloat,Ti};
    check::Bool=true,
) where {Ti<:Integer}
    # Preflight preserves an existing good factor when the ambient/input
    # precision or frozen pattern is wrong.
    _ambient_guard(cache.precision_bits, "factorize!")
    _validate_numeric(cache, A)
    cache.status = BFLA.FactorStatus(:unprepared, nothing)
    try
        BFLA.copy_owned!(cache.matrix.nzval, A.nzval)
        factor = something(cache.factor)
        QDLDL.update_values!(factor, cache.indices, cache.matrix.nzval)
        QDLDL.refactor!(factor)
        _factor_precision_ok(factor, cache.precision_bits) || throw(ArgumentError(
            "QDLDL numeric factor does not preserve BigFloat precision",
        ))
        cache.numeric_factor_count += 1
        cache.status = BFLA.SUCCESS_STATUS
    catch
        cache.status = BFLA.FactorStatus(:pivot_failure, nothing)
        check && rethrow()
    end
    return cache
end

function _solve_inplace!(cache::BFLASparseLDLCache, destination)
    factor = something(cache.factor)
    if ndims(destination) == 1
        QDLDL.solve!(factor, destination)
        solved = 1
    else
        @inbounds for column in axes(destination, 2)
            QDLDL.solve!(factor, view(destination, :, column))
        end
        solved = size(destination, 2)
    end
    _validate_values(destination, cache.precision_bits, "solve!")
    cache.solve_count += solved
    return destination
end

function _solve_preflight(cache::BFLASparseLDLCache, destination, rhs, op)
    _ambient_guard(cache.precision_bits, op)
    BFLA._cache_require_success(cache, op)
    size(destination) == size(rhs) || throw(DimensionMismatch(
        "$op: destination/RHS dimensions differ",
    ))
    size(rhs, 1) == cache.n || throw(DimensionMismatch(
        "$op: RHS row count differs from the factor order",
    ))
    nrhs = ndims(rhs) == 1 ? 1 : size(rhs, 2)
    nrhs <= cache.nrhs_capacity || throw(DimensionMismatch(
        "$op: RHS width exceeds prepared capacity",
    ))
    return nothing
end

function BFLA.solve_trusted!(
    destination::AbstractVecOrMat{BigFloat}, cache::BFLASparseLDLCache,
    rhs::AbstractVecOrMat{BigFloat},
)
    _solve_preflight(cache, destination, rhs, "solve_trusted!")
    BFLA._cache_rhs_write!(
        destination, rhs, cache.precision_bits, "solve_trusted!",
    )
    return _solve_inplace!(cache, destination)
end

function BFLA.solve!(
    destination::AbstractVecOrMat{BigFloat}, cache::BFLASparseLDLCache,
    rhs::AbstractVecOrMat{BigFloat},
)
    _solve_preflight(cache, destination, rhs, "solve!")
    BFLA._cache_rhs_repair!(destination, rhs, "solve!")
    return _solve_inplace!(cache, destination)
end

function BFLA.factor_diagnostics(cache::BFLASparseLDLCache)
    factor = cache.factor
    return (
        provider=:qdldl, kind=:sparse_ldlt,
        precision_bits=cache.precision_bits,
        pattern_signature=cache.pattern_signature,
        status=cache.status.kind, symbolic_count=cache.symbolic_count,
        numeric_factor_count=cache.numeric_factor_count,
        solve_count=cache.solve_count, nnz_k=nnz(cache.matrix),
        nnz_l=factor === nothing ? 0 : nnz(factor.L),
        positive_inertia=factor === nothing ? -1 : QDLDL.positive_inertia(factor),
        regularized_entries=factor === nothing ? 0 : QDLDL.regularized_entries(factor),
        julia_threads=Threads.nthreads(),
    )
end

end
