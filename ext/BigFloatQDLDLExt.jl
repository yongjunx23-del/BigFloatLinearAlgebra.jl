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
    dsigns::Vector{Ti}             # diagnostic only; never fed to QDLDL
    frozen_colptr::Vector{Ti}
    frozen_rowval::Vector{Ti}
    factored_values::Vector{BigFloat}
    factor_values_valid::Bool
    pattern_signature::UInt64
    nrhs_capacity::Int
    n::Int
    precision_bits::Int
    status::BFLA.FactorStatus
    prepared::Bool
    symbolic_count::Int
    numeric_factor_count::Int
    solve_count::Int
    positive_inertia::Int
    regularized_entries::Int
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
) where {Ti<:Integer}
    _ambient_guard(precision_bits, "sparse_ldlt_cache")
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    _validate_pattern(pattern, dsigns, precision_bits)
    frozen_colptr = copy(pattern.colptr)
    frozen_rowval = copy(pattern.rowval)
    values = BFLA.owned_copy(pattern.nzval; precision_bits=precision_bits)
    matrix = SparseMatrixCSC(
        size(pattern, 1), size(pattern, 2), copy(frozen_colptr),
        copy(frozen_rowval), values,
    )
    signs = Ti[sign for sign in dsigns]
    indices = Ti.(eachindex(matrix.nzval))
    # Dynamic regularization is deliberately disabled. Separate, unexposed,
    # precision-owned zeros prevent caller mutation from changing factor policy.
    eps_value = BigFloat(0; precision=precision_bits)
    delta_value = BigFloat(0; precision=precision_bits)
    factor = QDLDL.qdldl(
        matrix; logical=true, Dsigns=nothing,
        regularize_eps=eps_value, regularize_delta=delta_value,
    )
    _factor_precision_ok(factor, precision_bits) || throw(ArgumentError(
        "QDLDL symbolic workspace does not preserve BigFloat precision",
    ))
    n = size(matrix, 1)
    return BFLASparseLDLCache{Ti}(
        QDLDLBackend(), matrix, factor, indices, signs,
        frozen_colptr, frozen_rowval,
        BFLA.owned_copy(matrix.nzval; precision_bits=precision_bits), false,
        _pattern_signature(matrix, signs), Int(nrhs), n, precision_bits,
        BFLA.FactorStatus(:unprepared, nothing), true, 1, 0, 0, -1, 0,
    )
end

BFLA.factor_kind(::BFLASparseLDLCache) = :sparse_ldlt
function BFLA.factor_matrix(cache::BFLASparseLDLCache{Ti}) where {Ti}
    return SparseMatrixCSC(
        cache.n, cache.n, copy(cache.frozen_colptr), copy(cache.frozen_rowval),
        BFLA.owned_copy(cache.matrix.nzval; precision_bits=cache.precision_bits),
    )
end

function _validate_authority(cache::BFLASparseLDLCache)
    size(cache.matrix) == (cache.n, cache.n) || throw(ArgumentError(
        "QDLDL sparse LDL cache dimensions drifted from frozen authority",
    ))
    cache.matrix.colptr == cache.frozen_colptr &&
        cache.matrix.rowval == cache.frozen_rowval || throw(ArgumentError(
            "QDLDL sparse LDL internal pattern drift",
        ))
    _pattern_signature(cache.matrix, cache.dsigns) == cache.pattern_signature ||
        throw(ArgumentError("QDLDL sparse LDL pattern signature drift"))
    _validate_values(cache.matrix.nzval, cache.precision_bits, "cache authority")
    return nothing
end

function _validate_numeric(cache::BFLASparseLDLCache{Ti}, A) where {Ti}
    _validate_authority(cache)
    A isa SparseMatrixCSC{BigFloat,Ti} || throw(ArgumentError(
        "QDLDL sparse LDL numeric matrix must preserve BigFloat/index types",
    ))
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "QDLDL sparse LDL numeric dimensions differ from the frozen pattern",
    ))
    A.colptr == cache.frozen_colptr && A.rowval == cache.frozen_rowval ||
        throw(ArgumentError("QDLDL sparse LDL pattern drift"))
    _validate_values(A.nzval, cache.precision_bits, "factorize!")
    return nothing
end

function _revoke_factor!(cache::BFLASparseLDLCache)
    cache.status = BFLA.FactorStatus(:unprepared, nothing)
    cache.factor_values_valid = false
    cache.positive_inertia = -1
    cache.regularized_entries = 0
    return nothing
end

function BFLA.factorize!(
    cache::BFLASparseLDLCache{Ti}, A::SparseMatrixCSC{BigFloat,Ti};
    check::Bool=true,
) where {Ti<:Integer}
    # Every attempted factorization revokes the old solve authority before even
    # ambient/input preflight. A rejected input can never leave a Fresh factor.
    _revoke_factor!(cache)
    try
        _ambient_guard(cache.precision_bits, "factorize!")
        _validate_numeric(cache, A)
        BFLA.copy_owned!(cache.matrix.nzval, A.nzval)
        factor = something(cache.factor)
        # QDLDL update_values! assigns BigFloat object references. Feed it a
        # separate owned snapshot so its hidden triuA never aliases cache/input.
        provider_values = BFLA.owned_copy(
            cache.matrix.nzval; precision_bits=cache.precision_bits,
        )
        QDLDL.update_values!(factor, cache.indices, provider_values)
        QDLDL.refactor!(factor)
        _factor_precision_ok(factor, cache.precision_bits) || throw(ArgumentError(
            "QDLDL numeric factor does not preserve BigFloat precision",
        ))
        BFLA.copy_owned!(cache.factored_values, cache.matrix.nzval)
        cache.factor_values_valid = true
        cache.positive_inertia = QDLDL.positive_inertia(factor)
        cache.regularized_entries = QDLDL.regularized_entries(factor)
        cache.regularized_entries == 0 || throw(ArgumentError(
            "QDLDL unexpectedly regularized an entry",
        ))
        cache.numeric_factor_count += 1
        cache.status = BFLA.SUCCESS_STATUS
    catch
        _revoke_factor!(cache)
        cache.status = BFLA.FactorStatus(:pivot_failure, nothing)
        check && rethrow()
    end
    return cache
end

function _validate_solve_authority(cache::BFLASparseLDLCache)
    _validate_authority(cache)
    _validate_values(cache.factored_values, cache.precision_bits, "factor snapshot")
    cache.factor_values_valid && cache.matrix.nzval == cache.factored_values ||
        throw(ArgumentError("QDLDL sparse LDL numeric factor is stale"))
    return nothing
end

function _reject_factor_alias(cache::BFLASparseLDLCache, destination)
    factor = something(cache.factor)
    for storage in (
        cache.matrix.nzval, cache.factored_values, factor.L.nzval,
        factor.Dinv.diag, factor.workspace.D, factor.workspace.Dinv,
        factor.workspace.fwork, factor.workspace.Lx,
    )
        Base.mightalias(destination, storage) && throw(ArgumentError(
            "QDLDL sparse LDL destination must not alias factor storage",
        ))
    end
    return nothing
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
    _validate_solve_authority(cache)
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
    # Exact cache precision and finiteness are required before destination write.
    _validate_values(rhs, cache.precision_bits, op)
    _reject_factor_alias(cache, destination)
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
    success = BFLA.issuccess(cache)
    return (
        provider=:qdldl, kind=:sparse_ldlt,
        precision_bits=cache.precision_bits,
        pattern_signature=cache.pattern_signature,
        status=cache.status.kind, symbolic_count=cache.symbolic_count,
        numeric_factor_count=cache.numeric_factor_count,
        solve_count=cache.solve_count, nnz_k=nnz(cache.matrix),
        nnz_l=factor === nothing ? 0 : nnz(factor.L),
        positive_inertia=success ? cache.positive_inertia : -1,
        regularized_entries=success ? cache.regularized_entries : 0,
        julia_threads=Threads.nthreads(),
    )
end

end
