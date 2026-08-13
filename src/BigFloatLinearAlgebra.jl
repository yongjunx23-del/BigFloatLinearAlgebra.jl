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
include("level1.jl")
include("level2.jl")
include("level3.jl")
include("triangular.jl")
include("cholesky.jl")
include("generic_backend.jl")
include("native_backend.jl")

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
    :_gemm!,
    :_syrk!,
    :_trmm!,
    :_trsm!,
    :_cholesky!,
    :_cholesky_solve!,
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
    UnsupportedOperation,
    AbstractBFLAFactor,
    BFLACholeskyFactor,
    factor_matrix,
    factor_backend,
    factor_triangle,
    factor_precision,
    factor_status,
    issuccess,
    owned_zeros,
    owned_similar,
    owned_copy,
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
    gemm!,
    syrk!,
    trmm!,
    trsm!,
    mirror_triangle!,
    try_cholesky!,
    cholesky!,
    cholesky,
    ldiv!,
    solve!,
    solve

end
