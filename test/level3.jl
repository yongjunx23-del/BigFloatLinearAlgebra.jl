using Test
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

include("test_utils.jl")

function triangular_matrix(n::Int, p::Int, rng::AbstractRNG, triangle, diag)
    T = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in 1:n
        if triangle === Lower && i < j
            continue
        elseif triangle === Upper && i > j
            continue
        end
        if i == j
            T[i, j] = diag === UnitDiagonal ? BigFloat(1; precision = p) : BigFloat(n + i; precision = p)
        else
            T[i, j] = BigFloat(rand(rng); precision = p)
        end
    end
    return T
end

@testset "level3" begin
    for p in (128, 256, 512)
        rng = MersenneTwister(3000 + p)
        m, k, n = 5, 4, 3

        @testset "gemm! p=$p" begin
            for (transA, transB) in ((NoTrans, NoTrans), (NoTrans, Trans), (Trans, NoTrans), (Trans, Trans))
                Am = transA === NoTrans ? (m, k) : (k, m)
                Bm = transB === NoTrans ? (k, n) : (n, k)
                A = random_matrix(Am[1], Am[2], p, rng)
                B = random_matrix(Bm[1], Bm[2], p, rng)
                C0 = random_matrix(m, n, p, rng)
                a = random_scalar(p, rng)
                b = random_scalar(p, rng)
                Cn = BFLA.owned_copy(C0)
                BFLA.gemm!(Native, transA, transB, a, A, B, b, Cn)
                Cg = BFLA.owned_copy(C0)
                BFLA.gemm!(Generic, transA, transB, a, A, B, b, Cg)
                assert_close(Cn, Cg, p; label = "gemm!")
            end
            # one high-precision oracle comparison
            A = random_matrix(m, k, p, rng)
            B = random_matrix(k, n, p, rng)
            C0 = random_matrix(m, n, p, rng)
            a = random_scalar(p, rng)
            b = random_scalar(p, rng)
            Cn = BFLA.owned_copy(C0)
            BFLA.gemm!(Native, NoTrans, NoTrans, a, A, B, b, Cn)
            q = 2p
            A2 = BFLA.owned_copy(A; precision_bits = q)
            B2 = BFLA.owned_copy(B; precision_bits = q)
            C2 = BFLA.owned_copy(C0; precision_bits = q)
            BFLA.gemm!(Generic, NoTrans, NoTrans, BigFloat(a; precision = q), A2, B2, BigFloat(b; precision = q), C2)
            assert_close(Cn, round_precision(C2, p), p; label = "gemm! ref")
        end

        @testset "syrk! p=$p" begin
            for triangle in (Lower, Upper), trans in (NoTrans, Trans)
                r, c = 6, 5
                Am = trans === NoTrans ? (c, r) : (r, c)
                A = random_matrix(Am[1], Am[2], p, rng)
                C0 = random_matrix(c, c, p, rng)
                a = random_scalar(p, rng)
                b = random_scalar(p, rng)
                Cn = BFLA.owned_copy(C0)
                BFLA.syrk!(Native, triangle, trans, a, A, b, Cn)
                Cg = BFLA.owned_copy(C0)
                BFLA.syrk!(Generic, triangle, trans, a, A, b, Cg)
                # compare only the requested triangle
                for j in 1:c
                    rng_i = triangle === Lower ? (j:c) : (1:j)
                    for i in rng_i
                        @test Cn[i, j] == Cg[i, j]
                    end
                end
            end
        end

        @testset "trmm! p=$p" begin
            n, r = 5, 4
            for side in (LeftSide, RightSide), triangle in (Lower, Upper), trans in (NoTrans, Trans)
                T = triangular_matrix(n, p, rng, triangle, NonUnitDiagonal)
                B0 = side === LeftSide ? random_matrix(n, r, p, rng) : random_matrix(r, n, p, rng)
                a = random_scalar(p, rng)
                Bn = BFLA.owned_copy(B0)
                BFLA.trmm!(Native, side, triangle, trans, NonUnitDiagonal, a, T, Bn)
                Bg = BFLA.owned_copy(B0)
                BFLA.trmm!(Generic, side, triangle, trans, NonUnitDiagonal, a, T, Bg)
                assert_close(Bn, Bg, p; label = "trmm!")
            end
        end

        @testset "trsm! p=$p" begin
            n, r = 5, 4
            for side in (LeftSide, RightSide), triangle in (Lower, Upper), trans in (NoTrans, Trans)
                T = triangular_matrix(n, p, rng, triangle, NonUnitDiagonal)
                B0 = side === LeftSide ? random_matrix(n, r, p, rng) : random_matrix(r, n, p, rng)
                a = random_scalar(p, rng)
                Bn = BFLA.owned_copy(B0)
                BFLA.trsm!(Native, side, triangle, trans, NonUnitDiagonal, a, T, Bn)
                Bg = BFLA.owned_copy(B0)
                BFLA.trsm!(Generic, side, triangle, trans, NonUnitDiagonal, a, T, Bg)
                assert_close(Bn, Bg, p; label = "trsm!")
                # backward error check for the Native solve
                R = BFLA.owned_zeros(BigFloat, size(B0)...; precision_bits = p)
                if side === LeftSide
                    BFLA.gemm!(Native, trans, NoTrans, BigFloat(1; precision = p), T, Bn, BigFloat(0; precision = p), R)
                else
                    BFLA.gemm!(Native, NoTrans, trans, BigFloat(1; precision = p), Bn, T, BigFloat(0; precision = p), R)
                end
                for i in eachindex(R)
                    R[i] = R[i] - a * B0[i]
                end
                @test Float64(BFLA.norminf(Native, R)) < 1e-20
            end
        end
    end
end
