# Symmetric-indefinite LDLᵀ (Bunch-Kaufman) factorization and solves.

"""
    BFLALDLTFactor{M,B} <: AbstractBFLAFactor

Symmetric-indefinite `P A Pᵀ = L D Lᵀ` factorization, where `L` is unit lower
triangular and `D` is block diagonal with 1×1 and 2×2 blocks.

`factors` stores the packed lower factorization: unit diagonal and, below it,
the `L` multipliers plus `D`'s diagonal/subdiagonal on the 1×1/2×2 pivot blocks.
The authoritative triangle is lower; the upper triangle is not read.

  * `backend`: producing backend.
  * `precision_bits`: MPFR precision of the factor storage.
  * `status`: [`FactorStatus`](@ref) describing the result.
  * `perm`: permutation such that `perm[i]` is the original index at position `i`.
  * `blocks`: pivot block sizes (1 or 2) in order.
  * `subdiag_is_d`: whether row `i` is the second row of a 2×2 block (so
    `factors[i, i-1]` holds `D`'s off-diagonal, not an `L` multiplier).
"""
struct BFLALDLTFactor{M<:AbstractMatrix{BigFloat},B<:AbstractBFLABackend} <: AbstractBFLAFactor
    factors::M
    backend::B
    precision_bits::Int
    status::FactorStatus
    perm::Vector{Int}
    blocks::Vector{Int}
    subdiag_is_d::AbstractVector{Bool}
end

factor_matrix(F::BFLALDLTFactor) = F.factors
factor_backend(F::BFLALDLTFactor) = F.backend
factor_precision(F::BFLALDLTFactor) = F.precision_bits
factor_status(F::BFLALDLTFactor) = F.status
factor_kind(::BFLALDLTFactor) = :ldlt
factor_triangle(::BFLALDLTFactor) = Lower
issuccess(F::BFLALDLTFactor) = F.status.kind === :success

# Normalize a symmetric 2x2 block with positive row scales. The normalized
# determinant has the same sign and zero status as the mathematical
# determinant, without forming products such as d11*d22 or e*e at the original
# exponent scale.
function _ldlt_2x2_normalize_rows!(
    a::BigFloat,
    e1::BigFloat,
    e2::BigFloat,
    c::BigFloat,
    determinant::BigFloat,
    row_scale_1::BigFloat,
    row_scale_2::BigFloat,
    temporary_1::BigFloat,
    temporary_2::BigFloat,
    d11::BigFloat,
    e::BigFloat,
    d22::BigFloat,
)
    _factor_abs_to!(row_scale_1, d11)
    _factor_abs_to!(temporary_1, e)
    temporary_1 > row_scale_1 &&
        MA.operate_to!(row_scale_1, copy, temporary_1)
    MA.operate_to!(row_scale_2, copy, temporary_1)
    _factor_abs_to!(temporary_2, d22)
    temporary_2 > row_scale_2 &&
        MA.operate_to!(row_scale_2, copy, temporary_2)
    if iszero(row_scale_1) || iszero(row_scale_2)
        MA.operate!(zero, determinant)
        return false
    end

    _mpfr_div!(a, d11, row_scale_1)
    _mpfr_div!(e1, e, row_scale_1)
    _mpfr_div!(e2, e, row_scale_2)
    _mpfr_div!(c, d22, row_scale_2)
    MA.operate_to!(temporary_1, *, a, c)
    MA.operate_to!(temporary_2, *, e1, e2)
    MA.operate_to!(determinant, -, temporary_1, temporary_2)
    return !iszero(determinant)
end

function _ldlt_2x2_solve_normalized!(
    x1::BigFloat,
    x2::BigFloat,
    a::BigFloat,
    e1::BigFloat,
    e2::BigFloat,
    c::BigFloat,
    determinant::BigFloat,
    row_scale_1::BigFloat,
    row_scale_2::BigFloat,
    y1::BigFloat,
    y2::BigFloat,
    scaled_y1::BigFloat,
    scaled_y2::BigFloat,
    work::BigFloat,
)
    _mpfr_div!(scaled_y1, y1, row_scale_1)
    _mpfr_div!(scaled_y2, y2, row_scale_2)
    MA.operate_to!(x1, *, c, scaled_y1)
    MA.operate_to!(work, *, e1, scaled_y2)
    MA.operate!(-, x1, work)
    _mpfr_div!(x1, x1, determinant)
    MA.operate_to!(x2, *, a, scaled_y2)
    MA.operate_to!(work, *, e2, scaled_y1)
    MA.operate!(-, x2, work)
    _mpfr_div!(x2, x2, determinant)
    return x1, x2
end

"""
    factor_perm(F) -> Vector{Int}

Permutation of the LDLᵀ factorization; `perm[i]` is the original index at
position `i`.
"""
factor_perm(F::BFLALDLTFactor) = copy(F.perm)

"""
    factor_blocks(F) -> Vector{Int}

Pivot block sizes (1 or 2) in factorization order.
"""
factor_blocks(F::BFLALDLTFactor) = copy(F.blocks)

"""
    factor_inertia(F) -> (npos, nneg, nzero)

Number of positive, negative, and zero eigenvalues of the factor `D` (and
therefore of the original matrix, by Sylvester's law of inertia).
"""
function factor_inertia(F::BFLALDLTFactor)
    issuccess(F) || throw(ArgumentError(
        "factor_inertia: inertia is unavailable for an unsuccessful factor",
    ))
    p_actual = _require_precision(
        _check_precision(F.factors), "factor_inertia",
    )
    p_actual == F.precision_bits ||
        throw(PrecisionMismatch(F.precision_bits, p_actual, nothing))
    _triangle_finite(F.factors, Lower) || throw(DomainError(
        F.factors,
        "factor_inertia: authoritative factor triangle contains non-finite entries",
    ))
    return _factor_inertia_unchecked(F)
end

# Duck-typed over any factor-like object exposing `factors`, `blocks`, and
# `precision_bits` (the allocating factor and the reusable cache both do).
function _factor_inertia_unchecked(F)
    npos = 0
    nneg = 0
    nzero = 0
    A = F.factors
    scratch = [BigFloat(0; precision=F.precision_bits) for _ in 1:9]
    a, e1, e2, c, determinant, row_scale_1, row_scale_2,
        temporary_1, temporary_2 = scratch
    k = 1
    for s in F.blocks
        if s == 1
            d = A[k, k]
            if d > 0
                npos += 1
            elseif d < 0
                nneg += 1
            else
                nzero += 1
            end
            k += 1
        else
            d11 = A[k, k]
            e = A[k + 1, k]
            d22 = A[k + 1, k + 1]
            _ldlt_2x2_normalize_rows!(
                a, e1, e2, c, determinant, row_scale_1, row_scale_2,
                temporary_1, temporary_2, d11, e, d22,
            )
            if determinant < 0
                npos += 1
                nneg += 1
            elseif determinant > 0
                d11 > 0 ? (npos += 2) : (nneg += 2)
            else
                # singular 2×2 block: one nonzero eigenvalue plus one zero
                (d11 > 0 || d22 > 0) ? (npos += 1) : (nneg += 1)
                nzero += 1
            end
            k += 2
        end
    end
    return (npos, nneg, nzero)
end

function _ldlt_pivot_diagnostics(F::BFLALDLTFactor)
    _validate_factor_precision(F, "factor_diagnostics")
    _triangle_finite(F.factors, Lower) || throw(DomainError(
        F.factors,
        "factor_diagnostics: authoritative LDLT triangle contains " *
        "non-finite entries",
    ))
    return _ldlt_pivot_diagnostics_unchecked(F)
end

function _ldlt_pivot_diagnostics_unchecked(F::BFLALDLTFactor)
    p = F.precision_bits
    A = F.factors
    absolute = BigFloat(0; precision = p)
    product = BigFloat(0; precision = p)
    square = BigFloat(0; precision = p)
    determinant = BigFloat(0; precision = p)
    scale = BigFloat(0; precision = p)
    denominator = BigFloat(0; precision = p)
    quality = BigFloat(0; precision = p)
    normalized = [BigFloat(0; precision=p) for _ in 1:9]
    a, e1, e2, c, normalized_determinant, row_scale_1, row_scale_2,
        temporary_1, temporary_2 = normalized
    minimum_1x1 = nothing
    minimum_2x2_determinant = nothing
    minimum_2x2_quality = nothing
    k = 1
    @inbounds for block_size in F.blocks
        if block_size == 1
            _factor_abs_to!(absolute, A[k, k])
            if minimum_1x1 === nothing || absolute < minimum_1x1
                minimum_1x1 = MA.mutable_copy(absolute)
            end
            k += 1
            continue
        end

        _ldlt_2x2_normalize_rows!(
            a, e1, e2, c, normalized_determinant,
            row_scale_1, row_scale_2, temporary_1, temporary_2,
            A[k, k], A[k + 1, k], A[k + 1, k + 1],
        )
        _factor_abs_to!(absolute, normalized_determinant)
        # Preserve the established p-bit diagnostic trajectory when the
        # unscaled expression is representable; use the normalized form only
        # when direct intermediates overflow or become non-finite.
        MA.operate_to!(product, *, A[k, k], A[k + 1, k + 1])
        MA.operate_to!(square, *, A[k + 1, k], A[k + 1, k])
        MA.operate_to!(determinant, -, product, square)
        if isfinite(determinant)
            _factor_abs_to!(absolute, determinant)
        else
            _factor_abs_to!(absolute, normalized_determinant)
            MA.operate_to!(determinant, *, absolute, row_scale_1)
            MA.operate!(*, determinant, row_scale_2)
            MA.operate_to!(absolute, copy, determinant)
        end
        if minimum_2x2_determinant === nothing ||
           absolute < minimum_2x2_determinant
            minimum_2x2_determinant = MA.mutable_copy(absolute)
        end

        _factor_abs_to!(absolute, normalized_determinant)

        MA.operate!(zero, scale)
        for value in (A[k, k], A[k + 1, k], A[k + 1, k + 1])
            _factor_abs_to!(determinant, value)
            determinant > scale && MA.operate_to!(scale, copy, determinant)
        end
        if iszero(scale)
            MA.operate!(zero, quality)
        else
            _mpfr_div!(temporary_1, row_scale_1, scale)
            _mpfr_div!(temporary_2, row_scale_2, scale)
            MA.operate_to!(denominator, *, temporary_1, temporary_2)
            MA.operate_to!(quality, *, absolute, denominator)
        end
        if minimum_2x2_quality === nothing || quality < minimum_2x2_quality
            minimum_2x2_quality = MA.mutable_copy(quality)
        end
        k += 2
    end
    return minimum_1x1, minimum_2x2_determinant, minimum_2x2_quality
end

"""
    factor_diagnostics(F) -> NamedTuple

Numerical facts about the factorization: inertia, 1×1/2×2 pivot counts, and the
optional failure position. BFLA reports facts and does not decide what a caller
should do with them.
"""
function factor_diagnostics(F::BFLALDLTFactor)
    if issuccess(F)
        _validate_factor_precision(F, "factor_diagnostics")
        _triangle_finite(F.factors, Lower) || throw(DomainError(
            F.factors,
            "factor_diagnostics: authoritative LDLT triangle contains " *
            "non-finite entries",
        ))
    end
    inertia = issuccess(F) ? _factor_inertia_unchecked(F) : nothing
    n1 = count(==(1), F.blocks)
    n2 = count(==(2), F.blocks)
    minimum_1x1, minimum_2x2_determinant, minimum_2x2_quality =
        issuccess(F) ? _ldlt_pivot_diagnostics_unchecked(F) :
        (nothing, nothing, nothing)
    return (
        factor_kind = factor_kind(F),
        inertia = inertia,
        pivot_1x1_count = n1,
        pivot_2x2_count = n2,
        failure_position = factor_failure_position(F),
        min_abs_1x1_pivot = minimum_1x1,
        min_abs_2x2_determinant = minimum_2x2_determinant,
        min_normalized_2x2_quality = minimum_2x2_quality,
    )
end

Base.size(F::BFLALDLTFactor) = size(F.factors)
Base.size(F::BFLALDLTFactor, dimension::Integer) =
    size(F.factors, dimension)
Base.eltype(::BFLALDLTFactor{M,B}) where {M,B} = BigFloat

# --- public API ---------------------------------------------------------

"""
    ldlt!(backend, A; check=true) -> BFLALDLTFactor

Factor a symmetric (possibly indefinite) matrix `A` in place using
Bunch-Kaufman pivoting, and borrow its storage. The lower triangle is
authoritative. Distinct authoritative lower entries must not share `BigFloat`
storage; an ownership violation is rejected before mutation. On failure,
`check=true` throws; `check=false` returns a factor with a non-success
[`FactorStatus`](@ref).
"""
function ldlt! end

function ldlt!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    check::Bool=true,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    _require_square(A, "ldlt!")
    p = _require_precision(_check_precision(A), "ldlt!")
    identity_buffer = _workspace_identity_buffer(
        workspace, workspace_worker, p, "ldlt!",
    )
    _require_independent_triangle_elements(
        A, Lower, "ldlt!", identity_buffer,
    )
    n = size(A, 1)
    if !_triangle_finite(A, Lower)
        check && throw(DomainError(
            A, "ldlt!: authoritative lower triangle contains non-finite entries",
        ))
        identity = collect(1:n)
        return BFLALDLTFactor(
            A,
            backend,
            p,
            FactorStatus(:nonfinite, nothing),
            identity,
            Int[],
            falses(n),
        )
    end
    info, perm, blocks = _ldlt!(backend, A, p)
    subdiag = _subdiag_is_d(blocks, size(A, 1))
    if !_triangle_finite(A, Lower)
        check && throw(DomainError(
            A, "ldlt!: factorization produced non-finite entries",
        ))
        return BFLALDLTFactor(
            A,
            backend,
            p,
            FactorStatus(:nonfinite, nothing),
            perm,
            blocks,
            subdiag,
        )
    end
    if info != 0
        check && throw(LinearAlgebra.SingularException(info))
        return BFLALDLTFactor(A, backend, p, FactorStatus(:pivot_failure, info), perm, blocks, subdiag)
    end
    return BFLALDLTFactor(A, backend, p, SUCCESS_STATUS, perm, blocks, subdiag)
end

"""
    try_ldlt!(backend, A) -> Union{BFLALDLTFactor,Nothing}

Like `ldlt!(backend, A; check=false)`, but return `nothing` instead of a failed
factor.
"""
function try_ldlt! end

function try_ldlt!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    F = ldlt!(
        backend,
        A;
        check=false,
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
    return issuccess(F) ? F : nothing
end

"""
    ldlt(backend, A; check=true) -> BFLALDLTFactor

Allocating LDLᵀ factorization; `A` is deep-copied first.
"""
function ldlt end

function ldlt(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    check::Bool=true,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    _require_square(A, "ldlt")
    p = _require_precision(_check_precision(A), "ldlt")
    _workspace_identity_buffer(workspace, workspace_worker, p, "ldlt")
    return ldlt!(
        backend,
        owned_copy(A; precision_bits=p);
        check=check,
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
end

function _subdiag_is_d(blocks::Vector{Int}, n::Int)
    out = falses(n)
    k = 1
    for s in blocks
        if s == 2
            out[k + 1] = true
        end
        k += s
    end
    return out
end

function _ldlt_ldiv!(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat},
    trusted::Bool,
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
    operation::AbstractString,
)
    issuccess(F) || throw(LinearAlgebra.SingularException(
        F.status.position === nothing ? 0 : F.status.position,
    ))
    n = size(F.factors, 1)
    size(rhs, 1) == n || throw(DimensionMismatch(
        "$operation: right-hand side dimensions differ",
    ))
    _require_no_alias(rhs, F.factors, operation)
    if trusted
        _validate_trusted_rhs_precision(F, operation, rhs)
    else
        _validate_factor_precision(F, operation, rhs)
        _validate_factor_metadata(F, operation)
        _triangle_finite(F.factors, Lower) || throw(DomainError(
            F.factors,
            "$operation: authoritative factor triangle contains non-finite entries",
        ))
    end
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: right-hand side contains non-finite entries",
    ))
    _validate_solve_workspace(
        workspace, workspace_worker, F.precision_bits, operation,
    )
    _ldlt_solve!(F.backend, F, rhs, workspace, workspace_worker)
    _all_finite(rhs) || throw(DomainError(
        rhs, "$operation: solve produced non-finite entries",
    ))
    return rhs
end

function ldiv!(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _ldlt_ldiv!(F, rhs, false, workspace, workspace_worker, "ldiv!")
end

function ldiv_trusted!(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return _ldlt_ldiv!(
        F, rhs, true, workspace, workspace_worker, "ldiv_trusted!",
    )
end

function solve!(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return ldiv!(
        F, rhs; workspace=workspace, workspace_worker=workspace_worker,
    )
end

function solve(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat};
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    return ldiv!(
        F,
        owned_copy(rhs);
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
end

# --- Native Bunch-Kaufman -----------------------------------------------

@inline function _swap_sym!(A::AbstractMatrix{BigFloat}, r::Int, s::Int)
    r == s && return A
    n = size(A, 1)
    # These are one-for-one slot permutations, not copies: every MPFR object
    # still occupies exactly one array position after each swap.
    @inbounds for c in 1:n
        A[r, c], A[s, c] = A[s, c], A[r, c]
    end
    @inbounds for i in 1:n
        A[i, r], A[i, s] = A[i, s], A[i, r]
    end
    return A
end

function _bk_alpha(p::Int)
    a = BigFloat(0; precision = p)
    _mpfr_sqrt!(a, BigFloat(17; precision = p))
    MA.operate!(+, a, BigFloat(1; precision = p))
    _mpfr_div!(a, a, BigFloat(8; precision = p))
    return a
end

function _ldlt!(::NativeBackend, A::AbstractMatrix{BigFloat}, p::Int)
    n = size(A, 1)
    n == 0 && return 0, Int[], Int[]
    # Pivot selection needs symmetric row/column access. Rebuild the inactive
    # upper triangle only after backend dispatch has resolved to a supported
    # implementation, so an unsupported backend never mutates the caller's
    # target before throwing.
    mirror_triangle!(A, Lower)
    alpha = _bk_alpha(p)
    perm = collect(1:n)
    blocks = Int[]
    w = BigFloat(0; precision = p)
    rmax = BigFloat(0; precision = p)
    acc = BigFloat(0; precision = p)
    detv = BigFloat(0; precision = p)
    a1v = BigFloat(0; precision = p)
    a2v = BigFloat(0; precision = p)
    absolute_value = BigFloat(0; precision = p)
    threshold = BigFloat(0; precision = p)
    normalized = [BigFloat(0; precision=p) for _ in 1:10]
    norm_a, norm_e1, norm_e2, norm_c, norm_det, norm_r1, norm_r2,
        norm_y1, norm_y2, norm_work = normalized
    comparison_scale = BigFloat(0; precision = p)
    comparison_left = BigFloat(0; precision = p)
    comparison_right = BigFloat(0; precision = p)
    k = 1
    @inbounds while k <= n
        # largest off-diagonal magnitude in column k below the diagonal
        MA.operate!(zero, w)
        r = 0
        for i in (k + 1):n
            MA.operate_to!(absolute_value, abs, A[i, k])
            if absolute_value > w
                MA.operate_to!(w, copy, absolute_value)
                r = i
            end
        end
        if iszero(w)
            if iszero(A[k, k])
                return k, perm, blocks
            end
            push!(blocks, 1)
            k += 1
            continue
        end
        s = 1
        MA.operate_to!(absolute_value, abs, A[k, k])
        MA.operate_to!(a1v, copy, absolute_value)
        MA.operate_to!(threshold, *, alpha, w)
        if absolute_value < threshold
            # refined Bunch-Kaufman pivot selection
            MA.operate!(zero, rmax)
            for j in k:n
                j == r && continue
                MA.operate_to!(absolute_value, abs, A[r, j])
                if absolute_value > rmax
                    MA.operate_to!(rmax, copy, absolute_value)
                end
            end
            # Standard Bunch-Kaufman intermediate test:
            # |a_kk| >= alpha * colmax^2 / rowmax. Compare products at factor
            # precision to avoid a division and ambient precision dependence.
            MA.operate_to!(comparison_scale, copy, a1v)
            rmax > comparison_scale &&
                MA.operate_to!(comparison_scale, copy, rmax)
            w > comparison_scale &&
                MA.operate_to!(comparison_scale, copy, w)
            _mpfr_div!(comparison_left, a1v, comparison_scale)
            _mpfr_div!(comparison_right, rmax, comparison_scale)
            MA.operate!(*, comparison_left, comparison_right)
            _mpfr_div!(comparison_right, w, comparison_scale)
            MA.operate!(*, comparison_right, comparison_right)
            MA.operate!(*, comparison_right, alpha)
            if comparison_left >= comparison_right
                s = 1
            else
                MA.operate_to!(absolute_value, abs, A[r, r])
                MA.operate_to!(threshold, *, alpha, rmax)
                if absolute_value >= threshold
                    if r != k
                        _swap_sym!(A, k, r)
                        perm[k], perm[r] = perm[r], perm[k]
                    end
                    s = 1
                else
                    if r != k + 1
                        _swap_sym!(A, k + 1, r)
                        perm[k + 1], perm[r] = perm[r], perm[k + 1]
                    end
                    s = 2
                end
            end
        end

        if s == 1
            d = A[k, k]
            iszero(d) && return k, perm, blocks
            for i in (k + 1):n
                _mpfr_div!(A[i, k], A[i, k], d)
                MA.operate_to!(A[k, i], copy, A[i, k])
            end
            for j in (k + 1):n
                for i in j:n
                    # A[i,j] -= A[i,k] * d * A[j,k]
                    MA.operate_to!(w, *, A[i, k], d)
                    MA.operate_to!(rmax, *, w, A[j, k])
                    MA.operate_to!(A[i, j], -, A[i, j], rmax)
                    MA.operate_to!(A[j, i], copy, A[i, j])
                end
            end
            push!(blocks, 1)
            k += 1
        else
            e = A[k + 1, k]
            d11 = A[k, k]
            d22 = A[k + 1, k + 1]
            _ldlt_2x2_normalize_rows!(
                norm_a, norm_e1, norm_e2, norm_c, norm_det,
                norm_r1, norm_r2, norm_y1, norm_y2, d11, e, d22,
            ) || return k, perm, blocks
            for i in (k + 2):n
                # Copy the pivot-row entries into scratch before overwriting
                # A[i,k]/A[i,k+1] in place (BigFloat assignment aliases the
                # mutable object, so reading A[i,k] after the in-place divide
                # would observe the overwritten multiplier).
                MA.operate_to!(a1v, copy, A[i, k])
                MA.operate_to!(a2v, copy, A[i, k + 1])
                _ldlt_2x2_solve_normalized!(
                    A[i, k], A[i, k + 1],
                    norm_a, norm_e1, norm_e2, norm_c, norm_det,
                    norm_r1, norm_r2, a1v, a2v,
                    norm_y1, norm_y2, norm_work,
                )
                MA.operate_to!(A[k, i], copy, A[i, k])
                MA.operate_to!(A[k + 1, i], copy, A[i, k + 1])
            end
            for j in (k + 2):n
                for i in j:n
                    MA.operate_to!(w, *, A[i, k], d11)
                    MA.operate_to!(acc, *, w, A[j, k])
                    MA.operate_to!(w, *, A[i, k], A[j, k + 1])
                    MA.operate_to!(rmax, *, A[i, k + 1], A[j, k])
                    MA.operate!(+, w, rmax)
                    MA.operate!(*, w, e)
                    MA.operate!(+, acc, w)
                    MA.operate_to!(w, *, A[i, k + 1], d22)
                    MA.operate_to!(rmax, *, w, A[j, k + 1])
                    MA.operate!(+, acc, rmax)
                    MA.operate_to!(A[i, j], -, A[i, j], acc)
                    MA.operate_to!(A[j, i], copy, A[i, j])
                end
            end
            push!(blocks, 2)
            k += 2
        end
    end
    return 0, perm, blocks
end

_ldlt_solve!(
    ::NativeBackend,
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
) = _ldlt_solve_common!(F, rhs, workspace, workspace_worker)

_ldlt_solve!(
    ::GenericBackend,
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
) = _ldlt_solve_common!(F, rhs, workspace, workspace_worker)

function _ldlt_solve_common!(
    F,
    rhs::AbstractVecOrMat{BigFloat},
    workspace::Union{Nothing,BFLAWorkspace},
    workspace_worker::Int,
)
    n = size(F.factors, 1)
    p = F.precision_bits
    L = F.factors
    perm = F.perm
    subdiag = F.subdiag_is_d
    A = L
    acc = _solve_scratch(workspace, workspace_worker, 1, p)
    buf = _solve_scratch(workspace, workspace_worker, 2, p)
    storage = workspace === nothing ?
        owned_zeros(BigFloat, 4n + 10; precision_bits=p) :
        workspace_buffer!(workspace, workspace_worker, 4n + 10)
    z = view(storage, 1:n)
    y = view(storage, (n + 1):(2n))
    w = view(storage, (2n + 1):(3n))
    x = view(storage, (3n + 1):(4n))
    normalized = view(storage, (4n + 1):(4n + 10))
    a, e1, e2, c, scaled_determinant, row_scale_1, row_scale_2,
        scaled_y1, scaled_y2, solve_work = normalized
    @inbounds for col in axes(rhs, 2)
        # permute: z[i] = rhs[perm[i]]
        for i in 1:n
            MA.operate_to!(z[i], copy, rhs[perm[i], col])
        end
        # forward: L y = z
        for i in 1:n
            MA.operate!(zero, acc)
            for j in 1:(i - 1)
                (subdiag[i] && j == i - 1) && continue
                MA.buffered_operate!(buf, MA.add_mul, acc, A[i, j], y[j])
            end
            MA.operate_to!(y[i], -, z[i], acc)
        end
        # diagonal: D w = y
        k = 1
        for s in F.blocks
            if s == 1
                _mpfr_div!(w[k], y[k], A[k, k])
                k += 1
            else
                d11 = A[k, k]; e = A[k + 1, k]; d22 = A[k + 1, k + 1]
                _ldlt_2x2_normalize_rows!(
                    a, e1, e2, c, scaled_determinant,
                    row_scale_1, row_scale_2, scaled_y1, scaled_y2,
                    d11, e, d22,
                ) || throw(LinearAlgebra.SingularException(k))
                _ldlt_2x2_solve_normalized!(
                    w[k], w[k + 1], a, e1, e2, c, scaled_determinant,
                    row_scale_1, row_scale_2, y[k], y[k + 1],
                    scaled_y1, scaled_y2, solve_work,
                )
                k += 2
            end
        end
        # backward: Lᵀ z = w
        for i in n:-1:1
            MA.operate!(zero, acc)
            for j in (i + 1):n
                (subdiag[j] && i == j - 1) && continue
                MA.buffered_operate!(buf, MA.add_mul, acc, A[j, i], x[j])
            end
            MA.operate_to!(x[i], -, w[i], acc)
        end
        # unpermute: rhs[perm[i], col] = x[i]
        for i in 1:n
            MA.operate_to!(rhs[perm[i], col], copy, x[i])
        end
    end
    return rhs
end

# --- Generic reference --------------------------------------------------

function _ldlt!(::GenericBackend, A::AbstractMatrix{BigFloat}, p::Int)
    return _with_precision(p) do
        n = size(A, 1)
        n == 0 && return 0, Int[], Int[]
        mirror_triangle!(A, Lower)
        alpha = (BigFloat(1) + sqrt(BigFloat(17))) / BigFloat(8)
        perm = collect(1:n)
        blocks = Int[]
        normalized = [BigFloat(0; precision=p) for _ in 1:10]
        norm_a, norm_e1, norm_e2, norm_c, norm_det, norm_r1, norm_r2,
            norm_y1, norm_y2, norm_work = normalized
        y1copy = BigFloat(0; precision = p)
        y2copy = BigFloat(0; precision = p)
        k = 1
        while k <= n
            w = BigFloat(0)
            r = 0
            for i in (k + 1):n
                ai = abs(A[i, k])
                if ai > w
                    w = ai
                    r = i
                end
            end
            if w == 0
                if A[k, k] == 0
                    return k, perm, blocks
                end
                push!(blocks, 1)
                k += 1
                continue
            end
            s = 1
            absakk = abs(A[k, k])
            if absakk < alpha * w
                rmax = BigFloat(0)
                for j in k:n
                    j == r && continue
                    aj = abs(A[r, j])
                    aj > rmax && (rmax = aj)
                end
                comparison_scale = max(absakk, rmax, w)
                comparison_left = (absakk / comparison_scale) *
                    (rmax / comparison_scale)
                comparison_right = alpha * (w / comparison_scale)^2
                if comparison_left >= comparison_right
                    s = 1
                elseif abs(A[r, r]) >= alpha * rmax
                    if r != k
                        _swap_sym!(A, k, r)
                        perm[k], perm[r] = perm[r], perm[k]
                    end
                    s = 1
                else
                    if r != k + 1
                        _swap_sym!(A, k + 1, r)
                        perm[k + 1], perm[r] = perm[r], perm[k + 1]
                    end
                    s = 2
                end
            end
            if s == 1
                d = A[k, k]
                d == 0 && return k, perm, blocks
                for i in (k + 1):n
                    A[i, k] /= d
                    MA.operate_to!(A[k, i], copy, A[i, k])
                end
                for j in (k + 1):n
                    for i in j:n
                        A[i, j] -= A[i, k] * d * A[j, k]
                        MA.operate_to!(A[j, i], copy, A[i, j])
                    end
                end
                push!(blocks, 1)
                k += 1
            else
                e = A[k + 1, k]
                d11 = A[k, k]
                d22 = A[k + 1, k + 1]
                _ldlt_2x2_normalize_rows!(
                    norm_a, norm_e1, norm_e2, norm_c, norm_det,
                    norm_r1, norm_r2, norm_y1, norm_y2, d11, e, d22,
                ) || return k, perm, blocks
                for i in (k + 2):n
                    MA.operate_to!(y1copy, copy, A[i, k])
                    MA.operate_to!(y2copy, copy, A[i, k + 1])
                    _ldlt_2x2_solve_normalized!(
                        A[i, k], A[i, k + 1],
                        norm_a, norm_e1, norm_e2, norm_c, norm_det,
                        norm_r1, norm_r2, y1copy, y2copy,
                        norm_y1, norm_y2, norm_work,
                    )
                    MA.operate_to!(A[k, i], copy, A[i, k])
                    MA.operate_to!(A[k + 1, i], copy, A[i, k + 1])
                end
                for j in (k + 2):n
                    for i in j:n
                        A[i, j] -= (A[i, k] * d11 * A[j, k] +
                            (A[i, k] * A[j, k + 1] + A[i, k + 1] * A[j, k]) * e +
                            A[i, k + 1] * d22 * A[j, k + 1])
                        MA.operate_to!(A[j, i], copy, A[i, j])
                    end
                end
                push!(blocks, 2)
                k += 2
            end
        end
        return 0, perm, blocks
    end
end
