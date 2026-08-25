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
RRQR) and, for Cholesky, the authoritative triangle.
"""
function _validate_factor_shape(F, op::AbstractString)
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
    # Rank-policy scalars must be finite and non-negative.
    for (name, value) in (
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
    end
    return nothing
end
