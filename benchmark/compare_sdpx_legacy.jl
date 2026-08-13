# Opt-in development benchmark comparing BFLA Native against the frozen SDPX
# legacy BigFloat dense kernels. This is not a BFLA test dependency; it loads
# the SDPX kernel file by path. Set `SDPX_LEGACY_BIGFLOAT` to override the
# default sibling-path resolution.
#
# The SDPX legacy kernels rely on Julia's global `setprecision` for scratch
# objects, so every legacy call is wrapped in a scoped `setprecision(BigFloat, p)`
# while BFLA Native uses explicit target precision.

using Random
import MutableArithmetics as MA
using SparseArrays
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

using BigFloatLinearAlgebra:
    NoTrans,
    Trans,
    Lower,
    Upper,
    LeftSide,
    RightSide,
    UnitDiagonal,
    NonUnitDiagonal,
    issuccess,
    factor_matrix

function legacy_path()
    env = get(ENV, "SDPX_LEGACY_BIGFLOAT", "")
    isempty(env) || return env
    candidate = joinpath(@__DIR__, "..", "..", "SDPX", "SDPX-v041-unified-la-probes", "src", "kernels", "bigfloat.jl")
    isfile(candidate) || error("SDPX legacy kernel file not found at $candidate; set SDPX_LEGACY_BIGFLOAT")
    return candidate
end

include(legacy_path())

function owned_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(rand(rng, -1024:1024) // 1024; precision = p)
    end
    return A
end

println("=== SDPX legacy parity ===")
println("legacy: ", legacy_path())

for p in (128, 256, 512)
    rng = MersenneTwister(5000 + p)
    n = 8
    x = owned_matrix(n, 1, p, rng)[:, 1]
    y = owned_matrix(n, 1, p, rng)[:, 1]

    # dot
    d_sdpx = setprecision(BigFloat, p) do
        kdot(x, y)
    end
    d_bfla = BFLA.dot(Native, x, y)
    @assert d_sdpx == d_bfla "kdot parity failed at p=$p"

    # gemm (owned)
    A = owned_matrix(n, n, p, rng)
    B = owned_matrix(n, n, p, rng)
    Cb = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    Cs = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    one = BigFloat(1; precision = p)
    zero = BigFloat(0; precision = p)
    BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, Cb)
    setprecision(BigFloat, p) do
        kmul_owned!(Cs, A, B, one, zero)
    end
    @assert all(Cb[i, j] == Cs[i, j] for i in 1:n, j in 1:n) "kmul parity failed at p=$p"

    # cholesky + solve
    R = owned_matrix(n, n, p, rng)
    S = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    BFLA.gemm!(Native, Trans, NoTrans, one, R, R, zero, S)
    setprecision(BigFloat, p) do
        for i in 1:n
            S[i, i] = S[i, i] + one
        end
    end
    Lb = BFLA.owned_copy(S)
    Fb = BFLA.cholesky(Native, Lb)
    Ls = BFLA.owned_copy(S)
    ok = setprecision(BigFloat, p) do
        kchol!(Ls)
    end
    @assert ok && issuccess(Fb) "kchol parity failed at p=$p"
    # compare factor lower triangles
    for j in 1:n, i in j:n
        @assert factor_matrix(Fb)[i, j] == Ls[i, j] "factor parity failed at p=$p"
    end

    b = owned_matrix(n, 1, p, rng)[:, 1]
    xb = BFLA.owned_copy(b)
    BFLA.solve!(Fb, xb)
    xs = BFLA.owned_copy(b)
    setprecision(BigFloat, p) do
        kcholsolve_owned!(Ls, xs)
    end
    @assert all(xb[i] == xs[i] for i in 1:n) "solve parity failed at p=$p"

    println("p=$p parity OK (dot, gemm, cholesky, solve)")
end

println("=== parity passed ===")
