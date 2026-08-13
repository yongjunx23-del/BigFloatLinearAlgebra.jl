using Test
using Random
import LinearAlgebra
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
    factor_matrix,
    factor_backend,
    factor_triangle,
    factor_precision,
    factor_status,
    factor_kind,
    factor_diagnostics,
    factor_perm,
    factor_blocks,
    factor_inertia,
    factor_rank,
    factor_jpvt,
    factor_Rdiag,
    factor_tolerance,
    factor_pivots

eps_bits(p::Int) = BigFloat(2; precision = p)^(1 - p)

function random_scalar(p::Int, rng::AbstractRNG)
    return BigFloat(2 * rand(rng) - 1; precision = p)
end

function random_vector(n::Int, p::Int, rng::AbstractRNG)
    v = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
    for i in 1:n
        v[i] = BigFloat(2 * rand(rng) - 1; precision = p)
    end
    return v
end

function random_matrix(m::Int, n::Int, p::Int, rng::AbstractRNG)
    A = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    for j in 1:n, i in 1:m
        A[i, j] = BigFloat(2 * rand(rng) - 1; precision = p)
    end
    return A
end

"""
    is_independently_owned(A)

Reliable destructive-free ownership probe: mutable `BigFloat` objects are
independent iff no two slots share object identity.
"""
function is_independently_owned(A::AbstractArray{BigFloat})
    n = length(A)
    n <= 1 && return true
    for i in 1:n, j in (i + 1):n
        A[i] === A[j] && return false
    end
    return true
end

"""
    round_precision(x, p)

Round a `BigFloat` scalar or array to `p` bits.
"""
round_precision(x::BigFloat, p::Int) = BigFloat(x; precision = p)
round_precision(A::AbstractArray{BigFloat}, p::Int) = map(x -> BigFloat(x; precision = p), A)

function array_difference_norminf(A::AbstractArray{BigFloat}, B::AbstractArray{BigFloat})
    @assert size(A) == size(B)
    acc = BigFloat(0; precision = max(precision(first(A)), precision(first(B))))
    for i in eachindex(A, B)
        d = abs(A[i] - B[i])
        d > acc && (acc = d)
    end
    return acc
end

"""
    assert_close(A, B, p; label="")

Scaled relative error comparison. Uses `max(100 * n, 1) * eps(p)` as the
default tolerance, which tolerates summation-order differences while catching
precision loss or algorithmic errors.
"""
function assert_close(A::AbstractArray{BigFloat}, B::AbstractArray{BigFloat}, p::Int; label::AbstractString="", tol=nothing)
    size(A) == size(B) || error("assert_close: size mismatch in $label")
    na = BFLA.norminf(Native, A)
    nb = BFLA.norminf(Native, B)
    nd = array_difference_norminf(A, B)
    denom = na + nb
    if denom == 0
        @test nd == 0
        return
    end
    bound = tol === nothing ? max(100 * max(length(A), 1), 1) * eps_bits(p) : tol
    relerr = nd / denom
    @test relerr <= bound
    relerr <= bound || @info "assert_close failed" label relerr bound nd denom
end

"""
    make_spd(n, p; seed, delta_bits)

Deterministic SPD fixture `R' * R + delta * I`. `delta_bits` controls
conditioning: `nothing` gives a well-conditioned matrix, a positive value makes
`delta = 2^(-delta_bits)`.
"""
function make_spd(n::Int, p::Int; seed::Int=42, delta_bits=nothing)
    rng = MersenneTwister(seed)
    R = random_matrix(n, n, p, rng)
    A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    one_p = BigFloat(1; precision = p)
    BFLA.gemm!(Native, Trans, NoTrans, one_p, R, R, BigFloat(0; precision = p), A)
    delta = delta_bits === nothing ? one_p : BigFloat(2; precision = p)^(-delta_bits)
    setprecision(BigFloat, p) do
        for i in 1:n
            A[i, i] = A[i, i] + delta
        end
    end
    return A
end

"""
    one_p(p)

One at precision `p`, as a `BigFloat` (avoids allocating per assertion).
"""
one_p(p::Int) = BigFloat(1; precision = p)

"""
    reference_gemm(m, k, n, p, seed, transA, transB)

Evaluate the same GEMM at `2p` bits through the Generic backend and round back
to `p`, producing a higher-precision oracle.
"""
function reference_gemm(m::Int, k::Int, n::Int, p::Int, rng::AbstractRNG, transA, transB)
    Am = transA === NoTrans ? (m, k) : (k, m)
    Bm = transB === NoTrans ? (k, n) : (n, k)
    A = random_matrix(Am[1], Am[2], p, rng)
    B = random_matrix(Bm[1], Bm[2], p, rng)
    C0 = random_matrix(m, n, p, rng)
    alpha = random_scalar(p, rng)
    beta = random_scalar(p, rng)

    q = 2p
    A2 = BFLA.owned_copy(A; precision_bits = q)
    B2 = BFLA.owned_copy(B; precision_bits = q)
    C02 = BFLA.owned_copy(C0; precision_bits = q)
    alpha2 = BigFloat(alpha; precision = q)
    beta2 = BigFloat(beta; precision = q)
    Cref = BFLA.owned_zeros(BigFloat, m, n; precision_bits = q)
    BFLA.gemm!(Generic, transA, transB, alpha2, A2, B2, beta2, Cref)
    return A, B, C0, alpha, beta, round_precision(Cref, p)
end
