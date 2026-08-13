# Adversarial algebraic-property tests. These verify every triangular flag
# combination against the mathematical definition using an independent direct
# reference, rather than only comparing Native against Generic (which could
# share a conceptual mistake). Added in response to an external code review.

function triangular_matrix_adversarial(n::Int, p::Int, rng::AbstractRNG, triangle, diag)
    T = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        if triangle === Lower && i < j
            continue
        elseif triangle === Upper && i > j
            continue
        end
        if i == j
            # For UnitDiagonal, store deliberate garbage on the diagonal so a
            # correct kernel (which ignores the stored diagonal) is the only way
            # to pass; a kernel that reads the diagonal will fail.
            T[i, j] = diag === UnitDiagonal ? BigFloat(12345; precision = p) : BigFloat(n + i; precision = p)
        else
            T[i, j] = BigFloat(rand(rng); precision = p)
        end
    end
    return T
end

@inline function _coeff(A, i, j, trans, diag, one)
    i == j && diag === UnitDiagonal && return one
    return trans === NoTrans ? A[i, j] : A[j, i]
end

"""
    direct_matvec(opA, A, x, p)

Independent dense `op(A) * x` using a plain reduction loop. The result is
rounded at precision `p`.
"""
function direct_matvec(A::AbstractMatrix{BigFloat}, x::AbstractVector{BigFloat}, trans, diag, p::Int)
    m = trans === NoTrans ? size(A, 1) : size(A, 2)
    k = trans === NoTrans ? size(A, 2) : size(A, 1)
    y = BFLA.owned_zeros(BigFloat, m; precision_bits = p)
    acc = BigFloat(0; precision = p)
    one = BigFloat(1; precision = p)
    setprecision(BigFloat, p) do
        for i in 1:m
            acc = 0
            for l in 1:k
                acc += _coeff(A, i, l, trans, diag, one) * x[l]
            end
            y[i] = acc
        end
    end
    return y
end

"""
    direct_matmul(A, B, transA, transB, p)

Independent dense `op(A) * op(B)` using a plain reduction loop.
"""
function direct_matmul(A::AbstractMatrix{BigFloat}, B::AbstractMatrix{BigFloat}, transA, transB, diagA, diagB, p::Int)
    m = transA === NoTrans ? size(A, 1) : size(A, 2)
    k = transA === NoTrans ? size(A, 2) : size(A, 1)
    n = transB === NoTrans ? size(B, 2) : size(B, 1)
    C = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
    acc = BigFloat(0; precision = p)
    one = BigFloat(1; precision = p)
    setprecision(BigFloat, p) do
        for j in 1:n, i in 1:m
            acc = 0
            for l in 1:k
                acc += _coeff(A, i, l, transA, diagA, one) * _coeff(B, l, j, transB, diagB, one)
            end
            C[i, j] = acc
        end
    end
    return C
end

function residual_norminf(A, B)
    acc = BigFloat(0; precision = precision(first(A)))
    for i in eachindex(A, B)
        d = abs(A[i] - B[i])
        d > acc && (acc = d)
    end
    return acc
end

@testset "algebraic properties" begin
    for p in (128, 256)
        rng = MersenneTwister(9000 + p)
        tol = 100 * eps_bits(p)

        @testset "trsv! all flags p=$p" begin
            n = 9
            for triangle in (Lower, Upper), trans in (NoTrans, Trans), diag in (UnitDiagonal, NonUnitDiagonal)
                A = triangular_matrix_adversarial(n, p, rng, triangle, diag)
                b = random_vector(n, p, rng)
                b0 = BFLA.owned_copy(b)
                x = BFLA.owned_copy(b)
                BFLA.trsv!(Native, triangle, trans, diag, A, x)
                # independent residual: op(A) * x == b0
                Ax = direct_matvec(A, x, trans, diag, p)
                resid = residual_norminf(Ax, b0)
                @test resid <= tol * max(BFLA.norminf(Native, b0), one_p(p))
            end
        end

        @testset "trsm! all flags p=$p" begin
            n, r = 8, 5
            alpha = BigFloat(2; precision = p)
            for side in (LeftSide, RightSide), triangle in (Lower, Upper), trans in (NoTrans, Trans), diag in (UnitDiagonal, NonUnitDiagonal)
                A = triangular_matrix_adversarial(n, p, rng, triangle, diag)
                B0 = side === LeftSide ? random_matrix(n, r, p, rng) : random_matrix(r, n, p, rng)
                X = BFLA.owned_copy(B0)
                BFLA.trsm!(Native, side, triangle, trans, diag, alpha, A, X)
                # independent residual: op(A)*X == alpha*B0 (left) or X*op(A) == alpha*B0 (right)
                if side === LeftSide
                    AX = direct_matmul(A, X, trans, NoTrans, diag, NonUnitDiagonal, p)
                else
                    AX = direct_matmul(X, A, NoTrans, trans, NonUnitDiagonal, diag, p)
                end
                scaled = BFLA.owned_zeros(BigFloat, size(B0)...; precision_bits = p)
                setprecision(BigFloat, p) do
                    for i in eachindex(scaled)
                        scaled[i] = alpha * B0[i]
                    end
                end
                resid = residual_norminf(AX, scaled)
                @test resid <= tol * max(BFLA.norminf(Native, scaled), one_p(p))
            end
        end

        @testset "trmm! all flags p=$p" begin
            n, r = 8, 5
            alpha = BigFloat(2; precision = p)
            for side in (LeftSide, RightSide), triangle in (Lower, Upper), trans in (NoTrans, Trans), diag in (UnitDiagonal, NonUnitDiagonal)
                A = triangular_matrix_adversarial(n, p, rng, triangle, diag)
                B0 = side === LeftSide ? random_matrix(n, r, p, rng) : random_matrix(r, n, p, rng)
                X = BFLA.owned_copy(B0)
                BFLA.trmm!(Native, side, triangle, trans, diag, alpha, A, X)
                # independent reference: alpha*op(A)*B0 (left) or alpha*B0*op(A) (right)
                if side === LeftSide
                    ref = direct_matmul(A, B0, trans, NoTrans, diag, NonUnitDiagonal, p)
                else
                    ref = direct_matmul(B0, A, NoTrans, trans, NonUnitDiagonal, diag, p)
                end
                setprecision(BigFloat, p) do
                    for i in eachindex(ref)
                        ref[i] = alpha * ref[i]
                    end
                end
                assert_close(X, ref, p; label = "trmm!")
            end
        end

        @testset "syrk!/syr! triangle authority p=$p" begin
            # Independent reference for the requested triangle.
            alpha = BigFloat(2; precision = p)
            beta = BigFloat(3; precision = p)
            for triangle in (Lower, Upper), trans in (NoTrans, Trans)
                m, k = 7, 6
                A = random_matrix(m, k, p, rng)
                n = trans === NoTrans ? m : k
                kred = trans === NoTrans ? k : m
                C0 = random_matrix(n, n, p, rng)
                C = BFLA.owned_copy(C0)
                BFLA.syrk!(Native, triangle, trans, alpha, A, beta, C)
                Cref = BFLA.owned_copy(C0)
                setprecision(BigFloat, p) do
                    for j in 1:n
                        rngi = triangle === Lower ? (j:n) : (1:j)
                        for i in rngi
                            s = BigFloat(0; precision = p)
                            for l in 1:kred
                                ai = trans === NoTrans ? A[i, l] : A[l, i]
                                aj = trans === NoTrans ? A[j, l] : A[l, j]
                                s += ai * aj
                            end
                            Cref[i, j] = alpha * s + beta * C0[i, j]
                        end
                    end
                end
                # compare requested triangle only
                for j in 1:n
                    rngi = triangle === Lower ? (j:n) : (1:j)
                    for i in rngi
                        @test C[i, j] == Cref[i, j]
                    end
                end
            end
        end
    end
end
