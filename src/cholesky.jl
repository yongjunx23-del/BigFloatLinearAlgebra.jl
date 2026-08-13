# Cholesky factorization and solve public API.

"""
    cholesky!(backend, A; triangle=Lower, check=true) -> BFLACholeskyFactor

Factor the symmetric positive-definite matrix `A` in place and borrow its
storage. Only the requested triangle is read and written; the other triangle is
left untouched. On failure, `check=true` throws a `PosDefException`; with
`check=false` a factor with a nonzero `info` is returned. A failed
factorization may leave `A` partially overwritten.
"""
function cholesky! end

function cholesky!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
    check::Bool=true,
)
    _require_square(A, "cholesky!")
    _require_valid_triangle(triangle, "cholesky!")
    p = _require_precision(_check_precision(A), "cholesky!")
    if !_triangle_finite(A, triangle)
        check && throw(DomainError(
            A,
            "cholesky!: authoritative triangle contains non-finite entries",
        ))
        return BFLACholeskyFactor(A, backend, triangle, p, -1)
    end
    info = _cholesky!(backend, A, triangle, p)
    if info != 0
        check && throw(LinearAlgebra.PosDefException(info))
    end
    return BFLACholeskyFactor(A, backend, triangle, p, info)
end

"""
    try_cholesky!(backend, A; triangle=Lower) -> Union{BFLACholeskyFactor,Nothing}

Like `cholesky!(backend, A; check=false)`, but return `nothing` instead of a
failed factor. The factorization still mutates `A` in place.
"""
function try_cholesky! end

function try_cholesky!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
)
    F = cholesky!(backend, A; triangle=triangle, check=false)
    return issuccess(F) ? F : nothing
end

"""
    cholesky(backend, A; triangle=Lower, check=true) -> BFLACholeskyFactor

Allocating Cholesky factorization. `A` is deep-copied first, so the returned
factor owns its storage and the input is never modified.
"""
function cholesky end

function cholesky(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
    check::Bool=true,
)
    _require_square(A, "cholesky")
    p = _require_precision(_check_precision(A), "cholesky")
    B = owned_copy(A; precision_bits=p)
    return cholesky!(backend, B; triangle=triangle, check=check)
end

# Unpack a factor and dispatch the solve through its recorded backend.
_cholesky_solve!(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat}) =
    _cholesky_solve!(F.backend, F.factors, F.triangle, F.precision_bits, rhs)

"""
    ldiv!(factor, rhs) -> rhs

Solve the factorized system `factor.factors * factor.factors' * x = rhs` in
place, overwriting `rhs` with the solution. The solve dispatches through the
backend recorded in the factor.
"""
function ldiv!(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat})
    issuccess(F) || throw(LinearAlgebra.PosDefException(F.info))
    n = size(F.factors, 1)
    size(rhs, 1) == n ||
        throw(DimensionMismatch("ldiv!: right-hand side dimensions differ"))
    _require_precision(_check_precision(F.factors, rhs), "ldiv!")
    return _cholesky_solve!(F, rhs)
end

"""
    solve!(factor, rhs) -> rhs

Alias for `ldiv!(factor, rhs)`.
"""
function solve! end

solve!(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat}) = ldiv!(F, rhs)

"""
    solve(factor, rhs) -> Vector or Matrix

Allocating solve. `rhs` is deep-copied and the copy is solved in place, so the
caller's right-hand side is never modified.
"""
function solve end

function solve(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat})
    return ldiv!(F, owned_copy(rhs))
end
