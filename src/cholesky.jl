# Cholesky factorization and solve public API.

"""
    cholesky!(backend, A; triangle=Lower, check=true, workspace=nothing,
              workspace_worker=1) -> BFLACholeskyFactor

Factor the symmetric positive-definite matrix `A` in place and borrow its
storage. Only the requested triangle is read and written; the other triangle is
left untouched. On failure, `check=true` throws a `PosDefException`; with
`check=false` a factor with a nonzero `info` is returned. A failed
factorization may leave `A` partially overwritten. An explicit `workspace`
reuses only the worker-local identity buffer used by the ownership precheck;
its precision must match `A`, and concurrent calls must reserve distinct
`workspace_worker` slots.
"""
function cholesky! end

function cholesky!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    _require_square(A, "cholesky!")
    _require_valid_triangle(triangle, "cholesky!")
    p = _require_precision(_check_precision(A), "cholesky!")
    identity_buffer = _workspace_identity_buffer(
        workspace, workspace_worker, p, "cholesky!",
    )
    _require_independent_triangle_elements(
        A, triangle, "cholesky!", identity_buffer,
    )
    if !_triangle_finite(A, triangle)
        check && throw(DomainError(
            A,
            "cholesky!: authoritative triangle contains non-finite entries",
        ))
        return BFLACholeskyFactor(A, backend, triangle, p, FactorStatus(:nonfinite, nothing))
    end
    info = _cholesky_dispatch!(backend, A, triangle, p, config)
    if !_triangle_finite(A, triangle)
        check && throw(DomainError(
            A, "cholesky!: factorization produced non-finite entries",
        ))
        return BFLACholeskyFactor(
            A, backend, triangle, p, FactorStatus(:nonfinite, nothing),
        )
    end
    if info != 0
        check && throw(LinearAlgebra.PosDefException(info))
        return BFLACholeskyFactor(A, backend, triangle, p, FactorStatus(:not_positive_definite, info))
    end
    return BFLACholeskyFactor(A, backend, triangle, p, SUCCESS_STATUS)
end

"""
    try_cholesky!(backend, A; triangle=Lower, workspace=nothing,
                  workspace_worker=1) -> Union{BFLACholeskyFactor,Nothing}

Like `cholesky!(backend, A; check=false)`, but return `nothing` instead of a
failed factor. The factorization still mutates `A` in place.
"""
function try_cholesky! end

function try_cholesky!(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    F = cholesky!(
        backend,
        A;
        triangle=triangle,
        check=false,
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
    return issuccess(F) ? F : nothing
end

"""
    cholesky(backend, A; triangle=Lower, check=true, workspace=nothing,
             workspace_worker=1) -> BFLACholeskyFactor

Allocating Cholesky factorization. `A` is deep-copied first, so the returned
factor owns its storage and the input is never modified.
"""
function cholesky end

function cholesky(
    backend::AbstractBFLABackend,
    A::AbstractMatrix{BigFloat};
    triangle::Triangle=Lower,
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,BFLAWorkspace}=nothing,
    workspace_worker::Int=1,
)
    _require_square(A, "cholesky")
    p = _require_precision(_check_precision(A), "cholesky")
    _workspace_identity_buffer(workspace, workspace_worker, p, "cholesky")
    B = owned_copy(A; precision_bits=p)
    return cholesky!(
        backend,
        B;
        triangle=triangle,
        check=check,
        config=config,
        workspace=workspace,
        workspace_worker=workspace_worker,
    )
end

# Unpack a factor and dispatch the solve through its recorded backend.
_cholesky_solve!(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat}) =
    _cholesky_solve!(F.backend, F.factors, F.triangle, F.precision_bits, rhs)

"""
    ldiv!(factor, rhs) -> rhs

Solve the factorized system in place, overwriting `rhs` with the solution. For
a lower factor `L` this solves `(L * Lᵀ) * x = rhs`; for an upper factor `U`
(`GenericBackend` only) it solves `(Uᵀ * U) * x = rhs`. The solve dispatches
through the backend recorded in the factor.
"""
function ldiv!(F::BFLACholeskyFactor, rhs::AbstractVecOrMat{BigFloat})
    issuccess(F) || throw(LinearAlgebra.PosDefException(
        F.status.position === nothing ? 0 : F.status.position,
    ))
    n = size(F.factors, 1)
    size(rhs, 1) == n ||
        throw(DimensionMismatch("ldiv!: right-hand side dimensions differ"))
    _validate_factor_precision(F, "ldiv!", rhs)
    _require_no_alias(rhs, F.factors, "ldiv!")
    _triangle_finite(F.factors, F.triangle) || throw(DomainError(
        F.factors,
        "ldiv!: authoritative factor triangle contains non-finite entries",
    ))
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: right-hand side contains non-finite entries",
    ))
    _cholesky_solve!(F, rhs)
    _all_finite(rhs) || throw(DomainError(
        rhs, "ldiv!: solve produced non-finite entries",
    ))
    return rhs
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
