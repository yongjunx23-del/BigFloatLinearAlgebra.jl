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
issuccess(F::BFLALDLTFactor) = F.status.kind === :success

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
    npos = 0
    nneg = 0
    nzero = 0
    A = F.factors
    product = BigFloat(0; precision = F.precision_bits)
    square = BigFloat(0; precision = F.precision_bits)
    determinant = BigFloat(0; precision = F.precision_bits)
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
            MA.operate_to!(product, *, d11, d22)
            MA.operate_to!(square, *, e, e)
            MA.operate_to!(determinant, -, product, square)
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

"""
    factor_diagnostics(F) -> NamedTuple

Numerical facts about the factorization: inertia, 1×1/2×2 pivot counts, and the
optional failure position. BFLA reports facts and does not decide what a caller
should do with them.
"""
function factor_diagnostics(F::BFLALDLTFactor)
    inertia = issuccess(F) ? factor_inertia(F) : nothing
    n1 = count(==(1), F.blocks)
    n2 = count(==(2), F.blocks)
    return (
        inertia = inertia,
        pivot_1x1_count = n1,
        pivot_2x2_count = n2,
        failure_position = F.status.position,
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
)
    _require_square(A, "ldlt!")
    p = _require_precision(_check_precision(A), "ldlt!")
    _require_independent_triangle_elements(A, Lower, "ldlt!")
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

function try_ldlt!(backend::AbstractBFLABackend, A::AbstractMatrix{BigFloat})
    F = ldlt!(backend, A; check=false)
    return issuccess(F) ? F : nothing
end

"""
    ldlt(backend, A; check=true) -> BFLALDLTFactor

Allocating LDLᵀ factorization; `A` is deep-copied first.
"""
function ldlt end

function ldlt(backend::AbstractBFLABackend, A::AbstractMatrix{BigFloat}; check::Bool=true)
    _require_square(A, "ldlt")
    p = _require_precision(_check_precision(A), "ldlt")
    return ldlt!(backend, owned_copy(A; precision_bits=p); check=check)
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

function ldiv!(F::BFLALDLTFactor, rhs::AbstractVecOrMat{BigFloat})
    issuccess(F) || throw(LinearAlgebra.SingularException(
        F.status.position === nothing ? 0 : F.status.position,
    ))
    n = size(F.factors, 1)
    size(rhs, 1) == n || throw(DimensionMismatch("ldiv!: right-hand side dimensions differ"))
    _require_no_alias(rhs, F.factors, "ldiv!")
    p_actual = _require_precision(_check_precision(F.factors, rhs), "ldiv!")
    p_actual == F.precision_bits ||
        throw(PrecisionMismatch(F.precision_bits, p_actual, nothing))
    _triangle_finite(F.factors, Lower) || throw(DomainError(
        F.factors,
        "ldiv!: authoritative factor triangle contains non-finite entries",
    ))
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: right-hand side contains non-finite entries",
    ))
    _ldlt_solve!(F.backend, F, rhs)
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: solve produced non-finite entries",
    ))
    return rhs
end

solve!(F::BFLALDLTFactor, rhs::AbstractVecOrMat{BigFloat}) = ldiv!(F, rhs)
solve(F::BFLALDLTFactor, rhs::AbstractVecOrMat{BigFloat}) = ldiv!(F, owned_copy(rhs))

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
            MA.operate_to!(acc, *, a1v, rmax)
            MA.operate_to!(detv, *, w, w)
            MA.operate_to!(threshold, *, alpha, detv)
            if acc >= threshold
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
            MA.operate_to!(w, *, d11, d22)
            MA.operate_to!(rmax, *, e, e)
            MA.operate_to!(detv, -, w, rmax)
            iszero(detv) && return k, perm, blocks
            for i in (k + 2):n
                # Copy the pivot-row entries into scratch before overwriting
                # A[i,k]/A[i,k+1] in place (BigFloat assignment aliases the
                # mutable object, so reading A[i,k] after the in-place divide
                # would observe the overwritten multiplier).
                MA.operate_to!(a1v, copy, A[i, k])
                MA.operate_to!(a2v, copy, A[i, k + 1])
                MA.operate_to!(w, *, d22, a1v)
                MA.operate_to!(rmax, *, e, a2v)
                MA.operate_to!(acc, -, w, rmax)
                _mpfr_div!(A[i, k], acc, detv)
                MA.operate_to!(w, *, e, a1v)
                MA.operate_to!(rmax, *, d11, a2v)
                MA.operate_to!(acc, -, rmax, w)
                _mpfr_div!(A[i, k + 1], acc, detv)
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
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat},
) = _ldlt_solve_common!(F, rhs)

_ldlt_solve!(
    ::GenericBackend,
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat},
) = _ldlt_solve_common!(F, rhs)

function _ldlt_solve_common!(
    F::BFLALDLTFactor,
    rhs::AbstractVecOrMat{BigFloat},
)
    n = size(F.factors, 1)
    p = F.precision_bits
    L = F.factors
    perm = F.perm
    subdiag = F.subdiag_is_d
    A = L
    acc = BigFloat(0; precision = p)
    buf = BigFloat(0; precision = p)
    product = BigFloat(0; precision = p)
    square = BigFloat(0; precision = p)
    determinant = BigFloat(0; precision = p)
    numerator = BigFloat(0; precision = p)
    @inbounds for col in axes(rhs, 2)
        # permute: z[i] = rhs[perm[i]]
        z = [rhs[perm[i], col] for i in 1:n]
        y = [BigFloat(0; precision = p) for _ in 1:n]
        w = [BigFloat(0; precision = p) for _ in 1:n]
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
                MA.operate_to!(product, *, d11, d22)
                MA.operate_to!(square, *, e, e)
                MA.operate_to!(determinant, -, product, square)
                MA.operate_to!(product, *, d22, y[k])
                MA.operate_to!(square, *, e, y[k + 1])
                MA.operate_to!(numerator, -, product, square)
                _mpfr_div!(w[k], numerator, determinant)
                MA.operate_to!(product, *, e, y[k])
                MA.operate_to!(square, *, d11, y[k + 1])
                MA.operate_to!(numerator, -, square, product)
                _mpfr_div!(w[k + 1], numerator, determinant)
                k += 2
            end
        end
        # backward: Lᵀ z = w
        x = [BigFloat(0; precision = p) for _ in 1:n]
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
                if absakk * rmax >= alpha * w * w
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
                det = d11 * d22 - e * e
                det == 0 && return k, perm, blocks
                for i in (k + 2):n
                    a1 = A[i, k]; a2 = A[i, k + 1]
                    A[i, k] = (d22 * a1 - e * a2) / det
                    A[i, k + 1] = (-e * a1 + d11 * a2) / det
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
