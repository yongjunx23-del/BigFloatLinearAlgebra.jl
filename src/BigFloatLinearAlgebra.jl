"""
    BigFloatLinearAlgebra

Ownership-safe dense linear algebra for `BigFloat` / MPFR, with explicit
precision, Native and Generic backends, level 1–3 kernels, dense factorizations,
reusable factor caches, and a solver-independent residual/refinement API.
"""
module BigFloatLinearAlgebra

import LinearAlgebra
import LinearAlgebra: ldiv!, issuccess
using LinearAlgebra:
    Symmetric,
    LowerTriangular,
    UpperTriangular,
    UnitLowerTriangular,
    UnitUpperTriangular
import MutableArithmetics as MA

include("types.jl")
include("mpfr.jl")
include("validation.jl")
include("ownership.jl")
include("config.jl")
include("workspace.jl")
include("level1.jl")
include("level2.jl")
include("level3.jl")
include("triangular.jl")
include("cholesky.jl")
include("ldlt.jl")
include("qr.jl")
include("lu.jl")
include("quality.jl")
include("generic_backend.jl")
include("native_backend.jl")
include("caches.jl")

# These factories keep the optional LinearSolve integration discoverable from
# the parent package without importing either weak dependency in the core.
function _linearsolve_extension()
    extension = Base.get_extension(@__MODULE__, :BigFloatLinearSolveExt)
    extension === nothing && throw(ArgumentError(
        "BigFloatLU/BigFloatCholesky require LinearSolve and SciMLBase; " *
        "load both packages before constructing the algorithm",
    ))
    return extension
end

"""
    BigFloatLU(; backend=DEFAULT_BACKEND)

Construct the optional `LinearSolve.jl` LU algorithm. Load `LinearSolve` and
`SciMLBase` before calling this factory. The cache aliases `A` and `b`, while
BFLA's factor owns a deep matrix copy and never modifies either input. A
right-hand-side-only cache update reuses the factor; after mutating `A` in
place, use `cache.A = A` or `SciMLBase.reinit!(cache; A = A)` to refactor.
"""
function BigFloatLU(args...; kwargs...)
    return _linearsolve_extension().BigFloatLU(args...; kwargs...)
end

"""
    BigFloatCholesky(; backend=DEFAULT_BACKEND, triangle=Lower)

Construct the optional `LinearSolve.jl` Cholesky algorithm. Load `LinearSolve`
and `SciMLBase` before calling this factory. It has the same input-aliasing,
owned-factor, factor-reuse, and explicit matrix-refresh contract as
[`BigFloatLU`](@ref).
"""
function BigFloatCholesky(args...; kwargs...)
    return _linearsolve_extension().BigFloatCholesky(args...; kwargs...)
end

# A backend that is neither Native nor Generic must fail with an identifiable
# error rather than falling through to a different backend's kernel.
for name in (
    :_scal!,
    :_axpy!,
    :_axpby!,
    :_dot,
    :_norminf,
    :_gemv!,
    :_trsv!,
    :_syr!,
    :_symv!,
    :_gemm!,
    :_gemm_dispatch!,
    :_syrk!,
    :_syrk_dispatch!,
    :_gemmt!,
    :_syr2k!,
    :_trmm!,
    :_trsm!,
    :_trsm_dispatch!,
    :_cholesky!,
    :_cholesky_dispatch!,
    :_cholesky_solve!,
    :_ldlt!,
    :_ldlt_solve!,
    :_qr!,
    :_apply_q!,
    :_qr_solve!,
    :_lu!,
    :_lu_solve!,
    :_residual!,
    :_normwise_backward_error,
    :_higher_precision_residual!,
)
    @eval function $name(backend::AbstractBFLABackend, args...)
        _unsupported(backend, $(QuoteNode(name)), "no kernel registered for this backend")
    end
end

export AbstractBFLABackend,
    NativeBackend,
    GenericBackend,
    DEFAULT_BACKEND,
    TransposeOp,
    NoTrans,
    Trans,
    Triangle,
    Lower,
    Upper,
    Side,
    LeftSide,
    RightSide,
    DiagonalKind,
    UnitDiagonal,
    NonUnitDiagonal,
    capabilities,
    PrecisionMismatch,
    UnsupportedOperation,
    FactorStatus,
    KernelConfig,
    BFLAWorkspace,
    AbstractFactorCache,
    BFLACholeskyCache,
    BFLALUCache,
    BFLALDLTCache,
    BFLARRQRCache,
    prepare!,
    prepare_refinement!,
    factorize!,
    factor_prepared,
    factor_size,
    workspace_precision,
    workspace_workers,
    workspace_scratch!,
    workspace_buffer!,
    AbstractBFLAFactor,
    BFLACholeskyFactor,
    BFLALDLTFactor,
    BFLAQRFactor,
    BFLALUFactor,
    factor_matrix,
    factor_backend,
    factor_triangle,
    factor_precision,
    factor_status,
    factor_kind,
    factor_failure_position,
    factor_diagnostics,
    factor_perm,
    factor_blocks,
    factor_inertia,
    factor_rank,
    numerical_rank,
    factor_rank_atol,
    factor_rank_rtol,
    factor_rank_scale,
    factor_rank_threshold,
    factor_jpvt,
    factor_Rdiag,
    factor_tolerance,
    factor_pivots,
    issuccess,
    owned_zeros,
    owned_similar,
    owned_copy,
    convert_owned!,
    copy_owned!,
    zero_owned!,
    fill_owned!,
    scal!,
    axpy!,
    axpby!,
    dot,
    norminf,
    gemv!,
    trsv!,
    syr!,
    symv!,
    gemm!,
    syrk!,
    gemmt!,
    syr2k!,
    trmm!,
    trsm!,
    mirror_triangle!,
    try_cholesky!,
    cholesky!,
    cholesky,
    try_ldlt!,
    ldlt!,
    ldlt,
    qr!,
    qr,
    applyQ!,
    try_lu!,
    lu!,
    lu,
    BigFloatLU,
    BigFloatCholesky,
    residual!,
    normwise_backward_error,
    higher_precision_residual!,
    refinement_correction!,
    refine_once!,
    refine_once_trusted!,
    invalidate!,
    ldiv!,
    ldiv_trusted!,
    solve!,
    solve_trusted!,
    solve

end
