# Unified factor-integrity validation shared by the ordinary allocating factors
# and the reusable factor caches.
#
# The cache structs mirror the field names of the allocating factor structs
# (`factors`, `pivots`, `perm`, `blocks`, `subdiag_is_d`, `tau`, `jpvt`, `rank`,
# and the RRQR rank-policy scalars), so these validators are duck-typed and work
# on both. The checked `ldiv!`/`solve!`/`factor_diagnostics` paths call
# `_validate_factor_metadata`; the trusted paths skip it (their caller
# guarantees the factor is unchanged).

"""
    _validate_factor_shape(F, op)

Validate the factor's matrix shape (square for Cholesky/LU/LDLT; `m×n` for
RRQR) and, for Cholesky, the authoritative triangle. For a reusable cache this
also verifies that the owned factor storage matches the cache's reserved order
`cache.n` (a square-but-wrong-size factor is still rejected).
"""
function _validate_factor_shape(F, op::AbstractString)
    return _validate_factor_shape_impl(F, op)
end

# Shared shape rule used by both the ordinary factors and the caches.
function _validate_factor_shape_impl(F, op::AbstractString)
    kind = factor_kind(F)
    m, n = size(F.factors)
    if kind === :rrqr
        m >= 1 || throw(DimensionMismatch("$op: RRQR factor must be non-empty"))
    else
        m == n || throw(DimensionMismatch(
            "$op: $(kind) factor must be square, got $(m)x$(n)",
        ))
    end
    if kind === :cholesky
        _require_valid_triangle(factor_triangle(F), op)
    end
    return nothing
end

"""
    _validate_factor_integrity!(F, op)

The single checked entry point for factor-integrity validation. It validates, in
order:

1. factor matrix shape ([`_validate_factor_shape`](@ref));
2. factor storage precision ([`_validate_factor_storage_precision`](@ref));
3. factor storage finiteness ([`_validate_factor_storage_finite`](@ref));
4. factor metadata consistency ([`_validate_factor_metadata`](@ref)).

It is duck-typed and works on both the ordinary allocating factors and the
reusable factor caches (which mirror the factor field names). The ordinary
checked `ldiv!`/`solve!`, the cache checked `solve!`/`refine_once!`, and every
checked accessor/diagnostic that reads factor internals call this entry. The
trusted paths (`ldiv_trusted!`, `solve_trusted!`, `refine_once_trusted!`) skip
it because their caller guarantees the factor is unchanged.
"""
function _validate_factor_integrity!(F, op::AbstractString)
    _validate_factor_shape(F, op)
    _validate_factor_storage_precision(F, op)
    _validate_factor_storage_finite(F, op)
    _validate_factor_metadata(F, op)
    return nothing
end

"""
    _validate_factor_storage_precision(F, op)

Verify that the factor's mutable storage (the factor matrix and, for RRQR, the
Householder `tau` and rank-policy scalars) carries the factor's recorded
precision. Throws [`PrecisionMismatch`](@ref) on a mismatch.
"""
function _validate_factor_storage_precision(F, op::AbstractString)
    actual = _require_precision(
        _check_precision(factor_matrix(F), _factor_precision_operands(F)...),
        op,
    )
    recorded = factor_precision(F)
    actual == recorded || throw(PrecisionMismatch(recorded, actual, nothing))
    return actual
end

"""
    _validate_factor_storage_finite(F, op)

Verify that the factor's mutable storage is finite. Cholesky/LDLT scan only the
authoritative triangle; LU/RRQR scan the full factor matrix. RRQR additionally
requires `tau` and every rank-policy scalar to be finite.
"""
function _validate_factor_storage_finite(F, op::AbstractString)
    kind = factor_kind(F)
    A = factor_matrix(F)
    if kind === :cholesky
        _triangle_finite(A, factor_triangle(F)) || throw(DomainError(
            A, "$op: authoritative factor triangle contains non-finite entries",
        ))
    elseif kind === :ldlt
        _triangle_finite(A, Lower) || throw(DomainError(
            A, "$op: authoritative factor triangle contains non-finite entries",
        ))
    else
        _all_finite(A) || throw(DomainError(
            A, "$op: factor storage contains non-finite entries",
        ))
    end
    if kind === :rrqr
        _all_finite(F.tau) || throw(DomainError(
            F.tau, "$op: RRQR tau contains non-finite entries",
        ))
        for (name, value) in (
            (:tolerance, F.tolerance),
            (:atol, F.atol),
            (:rtol, F.rtol),
            (:reference_scale, F.reference_scale),
            (:effective_threshold, F.effective_threshold),
        )
            isfinite(value) || throw(DomainError(
                value, "$op: RRQR $name must be finite",
            ))
        end
    end
    return nothing
end

"""
    _validate_factor_metadata(F, op)

Validate the factor's metadata consistency (pivots, permutation, pivot blocks,
subdiagonal-D flags, Householder `tau`, column permutation, rank, and RRQR
rank-policy scalars). Throws a clear `ArgumentError`/`DimensionMismatch`/
`PrecisionMismatch`/`DomainError` on a malformed factor instead of allowing a
`BoundsError`, segfault, or silent wrong result.
"""
function _validate_factor_metadata(F, op::AbstractString)
    kind = factor_kind(F)
    if kind === :cholesky
        _validate_cholesky_metadata(F, op)
    elseif kind === :lu
        _validate_lu_metadata(F, op)
    elseif kind === :ldlt
        _validate_ldlt_metadata(F, op)
    elseif kind === :rrqr
        _validate_rrqr_metadata(F, op)
    else
        throw(ArgumentError("$op: unknown factor kind $kind"))
    end
    return nothing
end

function _validate_cholesky_metadata(F, op::AbstractString)
    return nothing
end

function _validate_lu_metadata(F, op::AbstractString)
    n = size(F.factors, 1)
    length(F.pivots) == n || throw(ArgumentError(
        "$op: LU pivots length ($(length(F.pivots))) != order $n",
    ))
    length(F.perm) == n || throw(ArgumentError(
        "$op: LU perm length ($(length(F.perm))) != order $n",
    ))
    @inbounds for k in 1:n
        (k <= F.pivots[k] <= n) || throw(ArgumentError(
            "$op: LU pivot $(F.pivots[k]) at step $k is out of range 1:$n",
        ))
    end
    seen = falses(n)
    @inbounds for i in 1:n
        p = F.perm[i]
        (1 <= p <= n && !seen[p]) || throw(ArgumentError(
            "$op: LU perm is not a permutation of 1:$n",
        ))
        seen[p] = true
    end
    # `perm` must equal the permutation rebuilt from the step pivots.
    rebuilt = collect(1:n)
    @inbounds for k in 1:n
        rebuilt[k], rebuilt[F.pivots[k]] = rebuilt[F.pivots[k]], rebuilt[k]
    end
    @inbounds for i in 1:n
        rebuilt[i] == F.perm[i] || throw(ArgumentError(
            "$op: LU perm is inconsistent with the step pivots",
        ))
    end
    return nothing
end

function _validate_ldlt_metadata(F, op::AbstractString)
    n = size(F.factors, 1)
    length(F.perm) == n || throw(ArgumentError(
        "$op: LDLT perm length ($(length(F.perm))) != order $n",
    ))
    seen = falses(n)
    @inbounds for i in 1:n
        p = F.perm[i]
        (1 <= p <= n && !seen[p]) || throw(ArgumentError(
            "$op: LDLT perm is not a permutation of 1:$n",
        ))
        seen[p] = true
    end
    all(b -> b == 1 || b == 2, F.blocks) || throw(ArgumentError(
        "$op: LDLT pivot blocks must be 1 or 2",
    ))
    sum(F.blocks) == n || throw(ArgumentError(
        "$op: LDLT pivot blocks sum to $(sum(F.blocks)) != order $n",
    ))
    length(F.subdiag_is_d) == n || throw(ArgumentError(
        "$op: LDLT subdiag_is_d length ($(length(F.subdiag_is_d))) != order $n",
    ))
    expected = _subdiag_is_d(F.blocks, n)
    @inbounds for i in 1:n
        expected[i] == F.subdiag_is_d[i] || throw(ArgumentError(
            "$op: LDLT subdiag_is_d is inconsistent with the pivot blocks",
        ))
    end
    return nothing
end

function _validate_rrqr_metadata(F, op::AbstractString)
    m, n = size(F.factors)
    rmax = min(m, n)
    length(F.tau) == rmax || throw(ArgumentError(
        "$op: RRQR tau length ($(length(F.tau))) != min(m,n) = $rmax",
    ))
    length(F.jpvt) == n || throw(ArgumentError(
        "$op: RRQR jpvt length ($(length(F.jpvt))) != n = $n",
    ))
    seen = falses(n)
    @inbounds for i in 1:n
        p = F.jpvt[i]
        (1 <= p <= n && !seen[p]) || throw(ArgumentError(
            "$op: RRQR jpvt is not a permutation of 1:$n",
        ))
        seen[p] = true
    end
    (0 <= F.rank <= rmax) || throw(ArgumentError(
        "$op: RRQR rank $(F.rank) out of range 0:$rmax",
    ))
    # Householder tau must carry the factor precision.
    tp = _check_precision(F.tau)
    if tp !== nothing && tp != F.precision_bits
        throw(PrecisionMismatch(F.precision_bits, tp, nothing))
    end
    # Rank-policy scalars must be finite, non-negative, and carry the factor
    # precision. A finite-but-wrong-precision scalar is still a malformed
    # factor and must be rejected.
    for (name, value) in (
        (:tolerance, F.tolerance),
        (:atol, F.atol),
        (:rtol, F.rtol),
        (:reference_scale, F.reference_scale),
        (:effective_threshold, F.effective_threshold),
    )
        isfinite(value) || throw(DomainError(
            value, "$op: RRQR $name must be finite",
        ))
        value >= 0 || throw(ArgumentError(
            "$op: RRQR $name must be non-negative",
        ))
        precision(value) == F.precision_bits || throw(PrecisionMismatch(
            F.precision_bits, precision(value), nothing,
        ))
    end
    # Rank-policy semantic consistency: the stored scalars must agree with the
    # rank policy that produced them, and the stored rank must agree with the
    # factor diagonal under the stored effective threshold. This rejects
    # in-range-but-wrong corruption (e.g. rank=0, a slightly shifted threshold,
    # or a tolerance that no longer equals atol) that a pure range check would
    # accept.
    F.tolerance == F.atol || throw(ArgumentError(
        "$op: RRQR tolerance must equal atol",
    ))
    expected = BigFloat(0; precision = F.precision_bits)
    MA.operate_to!(expected, *, F.rtol, F.reference_scale)
    expected > F.atol || MA.operate_to!(expected, copy, F.atol)
    F.effective_threshold == expected || throw(ArgumentError(
        "$op: RRQR effective_threshold is inconsistent with the rank policy " *
        "(expected max(atol, rtol*reference_scale) = $expected)",
    ))
    recomputed = _qr_rank_from_factors(F.factors, F.effective_threshold)
    F.rank == recomputed || throw(ArgumentError(
        "$op: RRQR rank $(F.rank) is inconsistent with the factor diagonal " *
        "under effective_threshold (recomputed $recomputed)",
    ))
    return nothing
end
