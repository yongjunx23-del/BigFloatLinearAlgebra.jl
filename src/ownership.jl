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
`precision_bits`, or the precision of the first source element when the source
is non-empty.
"""
function _owned_precision(A::AbstractArray{BigFloat}, precision_bits::Union{Nothing,Int})
    precision_bits === nothing || return precision_bits
    isempty(A) && throw(ArgumentError(
        "cannot infer precision from an empty array; pass precision_bits explicitly",
    ))
    return precision(first(A))
end

"""
    owned_similar(A; precision_bits) -> Array{BigFloat}

Allocate an owned zero array with the same shape as `A`. By default the
precision matches the first element of `A`; pass `precision_bits` to override.
"""
function owned_similar(A::AbstractArray{BigFloat}; precision_bits::Union{Nothing,Int}=nothing)
    p = _owned_precision(A, precision_bits)
    return owned_zeros(BigFloat, size(A)...; precision_bits = p)
end

"""
    owned_copy(A; precision_bits) -> Array{BigFloat}

Deep numeric copy of `A`. Every output element is an independent MPFR object,
so mutating the destination never changes the source and mutating one element
never changes another. By default the precision matches the source.
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

"""
    copy_owned!(destination, source) -> destination

Copy values into `destination` so that every destination element becomes an
independent MPFR object equal to the corresponding source element. The source
is never mutated. Source and destination element precisions must match.
"""
function copy_owned!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
)
    axes(destination) == axes(source) ||
        throw(DimensionMismatch("copy_owned!: arrays must have matching axes"))
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
`value`'s precision. Unlike `fill!(A, value)`, no two elements share storage.
"""
function fill_owned!(A::AbstractArray{BigFloat}, value::BigFloat)
    @inbounds for index in eachindex(A)
        A[index] = MA.mutable_copy(value)
    end
    return A
end
