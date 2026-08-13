# Ownership-safe storage. `zeros(BigFloat, ...)`, `fill(BigFloat(...), n)`,
# `copy`, and `Matrix(A)` do *not* guarantee independent MPFR storage: the first
# two install one shared mutable object in every slot, and the latter two copy
# object references rather than values. These helpers always install or
# preserve independent MPFR objects so in-place kernels have well-defined
# destinations.

"""
    owned_zeros(BigFloat, dims...; precision_bits) -> Array{BigFloat}

Create a zero-filled array whose every element owns an independent MPFR object
at exactly `precision_bits` bits. Unlike `zeros(BigFloat, ...)`, mutating one
element cannot affect another. `precision_bits` is required so long-lived
storage is never created from the ambient global precision.
"""
function owned_zeros(::Type{BigFloat}, dims::Integer...; precision_bits::Int)
    precision_bits > 0 ||
        throw(ArgumentError("precision_bits must be positive, got $(precision_bits)"))
    A = Array{BigFloat}(undef, dims...)
    @inbounds for index in eachindex(A)
        A[index] = BigFloat(0; precision = precision_bits)
    end
    return A
end

"""
    _owned_precision(A, precision_bits) -> Int

Resolve the target precision for an allocating copy: an explicit
`precision_bits`, or the uniform source precision when the source is non-empty.
The default path performs a full scan and fails closed with
[`PrecisionMismatch`](@ref) if the source elements are not all the same
precision, so a mixed-precision source is never silently normalized.
"""
function _owned_precision(A::AbstractArray{BigFloat}, precision_bits::Union{Nothing,Int})
    precision_bits === nothing || return precision_bits
    isempty(A) && throw(ArgumentError(
        "cannot infer precision from an empty array; pass precision_bits explicitly",
    ))
    p = _check_precision(A)
    return p === nothing ? throw(ArgumentError("could not determine precision")) : p
end

"""
    owned_similar(A; precision_bits) -> Array{BigFloat}

Allocate an owned zero array with the same shape as `A`. By default the
precision matches the uniform source precision (`PrecisionMismatch` on a mixed
source); pass `precision_bits` to override.
"""
function owned_similar(A::AbstractArray{BigFloat}; precision_bits::Union{Nothing,Int}=nothing)
    p = _owned_precision(A, precision_bits)
    return owned_zeros(BigFloat, size(A)...; precision_bits = p)
end

"""
    owned_copy(A; precision_bits) -> Array{BigFloat}

Deep numeric copy of `A`. Every output element is an independent MPFR object,
so mutating the destination never changes the source and mutating one element
never changes another.

By default (`precision_bits === nothing`) the source must have a uniform
precision; a mixed-precision source throws [`PrecisionMismatch`](@ref). Pass
`precision_bits = q` to explicitly round every source value into newly-owned
`q`-bit storage.
"""
function owned_copy(A::AbstractArray{BigFloat}; precision_bits::Union{Nothing,Int}=nothing)
    p = _owned_precision(A, precision_bits)
    B = owned_zeros(BigFloat, size(A)...; precision_bits = p)
    return _convert_copy!(B, A)
end

# Round each source value into the corresponding (possibly different-precision)
# destination object. Used by `owned_copy` to support explicit precision
# conversion; the public `copy_owned!` stays strict about matching precision.
function _convert_copy!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
)
    @inbounds for index in eachindex(destination, source)
        MA.operate_to!(destination[index], copy, source[index])
    end
    return destination
end

# `Base.mightalias` detects overlapping array storage, but two distinct arrays
# can still contain the same mutable MPFR objects after a shallow `copy`. This
# one-time identity scan is reserved for ownership conversion APIs; numerical
# kernels continue to require independently-owned destination storage.
function _require_no_shared_elements(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
    operation::AbstractString,
)
    source_elements = IdDict{BigFloat,Nothing}()
    @inbounds for value in source
        source_elements[value] = nothing
    end
    @inbounds for value in destination
        haskey(source_elements, value) && throw(ArgumentError(
            "$operation: destination must not share BigFloat objects with source",
        ))
    end
    return nothing
end

# Verify the destination-side ownership precondition for an in-place factor.
# This remains internal: object identity detects shared Julia BigFloat objects,
# but it is not exposed as a claim about arbitrary MPFR pointer ownership.
function _require_independent_triangle_elements(
    A::AbstractMatrix{BigFloat},
    triangle::Triangle,
    operation::AbstractString,
    identity_buffer::Union{Nothing,Vector{UInt}}=nothing,
)
    n = size(A, 1)
    required = n * (n + 1) ÷ 2
    object_ids = if identity_buffer === nothing
        Vector{UInt}(undef, required)
    else
        resize!(identity_buffer, required)
    end
    position = 0
    @inbounds for column in axes(A, 2), row in axes(A, 1)
        authoritative = triangle === Lower ? row >= column : row <= column
        authoritative || continue
        position += 1
        object_ids[position] = objectid(A[row, column])
    end
    sort!(object_ids; alg=Base.Sort.QuickSort)
    possible_duplicate = false
    @inbounds for index in 2:length(object_ids)
        if object_ids[index] == object_ids[index - 1]
            possible_duplicate = true
            break
        end
    end
    possible_duplicate || return nothing

    # `objectid` collisions are allowed. Only the uncommon duplicate-ID path
    # pays for an exact identity map, avoiding false rejection while keeping
    # the ordinary factorization path compact.
    elements = IdDict{BigFloat,Nothing}()
    @inbounds for column in axes(A, 2), row in axes(A, 1)
        authoritative = triangle === Lower ? row >= column : row <= column
        authoritative || continue
        value = A[row, column]
        haskey(elements, value) && throw(ArgumentError(
            "$operation: distinct authoritative entries must not share " *
            "BigFloat storage",
        ))
        elements[value] = nothing
    end
    return nothing
end

"""
    convert_owned!(destination, source) -> destination

Explicitly convert every `source` value into the precision already carried by
`destination`. Source element precisions may differ; numerical rounding into
the destination precision is intentional. The operation writes into the
existing destination MPFR objects, preserving their ownership and object
identity, and never mutates the source.

Source and destination must have matching axes and must not overlap. The
destination must be non-empty and uniformly precise. BFLA does not perform an
unreliable public ownership probe, so callers remain responsible for providing
independently-owned destination storage (for example from `owned_zeros`). Use
strict [`copy_owned!`](@ref) when no precision conversion is intended.
"""
function convert_owned!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "convert_owned!: arrays must have matching axes",
    ))
    _require_no_alias(destination, source, "convert_owned!")
    _require_no_shared_elements(destination, source, "convert_owned!")
    _require_precision(_check_precision(destination), "convert_owned!")
    return _convert_copy!(destination, source)
end

"""
    copy_owned!(destination, source) -> destination

Copy values into `destination` so that every destination element becomes an
independent MPFR object equal to the corresponding source element. The source
is never mutated. Source and destination element precisions must match, and
their array storage must not overlap. Distinct arrays that initially share
BigFloat objects are supported because every destination slot is replaced by a
newly-owned object.
"""
function copy_owned!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
)
    axes(destination) == axes(source) ||
        throw(DimensionMismatch("copy_owned!: arrays must have matching axes"))
    _require_no_alias(destination, source, "copy_owned!")
    p = _check_precision(destination, source)
    _require_precision(p, "copy_owned!")
    @inbounds for index in eachindex(destination, source)
        destination[index] = MA.mutable_copy(source[index])
    end
    return destination
end

"""
    zero_owned!(A) -> A

Reset every element of `A` to an independent zero object. This repairs aliased
storage (unlike an in-place reset), at the cost of one MPFR object per element.
"""
function zero_owned!(A::AbstractArray{BigFloat})
    p = _precision_bits(A)
    p === nothing && return A
    zero_value = BigFloat(0; precision = p)
    @inbounds for index in eachindex(A)
        A[index] = MA.mutable_copy(zero_value)
    end
    return A
end

"""
    fill_owned!(A, value) -> A

Fill every element of `A` with an independent copy of `value`, preserving
the uniform precision already carried by `A`. `value` must have that same
precision; cross-precision fill is rejected with [`PrecisionMismatch`](@ref)
before `A` is modified. Unlike `fill!(A, value)`, no two elements share
storage.
"""
function fill_owned!(A::AbstractArray{BigFloat}, value::BigFloat)
    _require_precision(_check_precision(A, value), "fill_owned!")
    @inbounds for index in eachindex(A)
        A[index] = MA.mutable_copy(value)
    end
    return A
end
