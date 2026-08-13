@testset "blocked kernels" begin
    for p in (128, 256)
        @testset "blocked p=$p" begin
            rng = MersenneTwister(9000 + p)
            a = BigFloat(2; precision = p)
            b = BigFloat(3; precision = p)
            one = BigFloat(1; precision = p)

            @testset "gemm blocked bit-parity" begin
                A = random_matrix(6, 5, p, rng)
                B = random_matrix(5, 4, p, rng)
                C0 = random_matrix(6, 4, p, rng)
                for transA in (NoTrans, Trans), transB in (NoTrans, Trans)
                    Am = transA === NoTrans ? (6, 5) : (5, 6)
                    Bm = transB === NoTrans ? (5, 4) : (4, 5)
                    A = random_matrix(Am[1], Am[2], p, rng)
                    B = random_matrix(Bm[1], Bm[2], p, rng)
                    C0 = random_matrix(6, 4, p, rng)
                    Cn = BFLA.owned_copy(C0)
                    BFLA.gemm!(Native, transA, transB, a, A, B, b, Cn; config = BFLA.KernelConfig(gemm_block = 3))
                    Cb = BFLA.owned_copy(C0)
                    BFLA.gemm!(Native, transA, transB, a, A, B, b, Cb)
                    @test all(Cn[i, j] == Cb[i, j] for i in 1:6, j in 1:4)
                end
            end

            @testset "syrk blocked bit-parity" begin
                for triangle in (Lower, Upper), trans in (NoTrans, Trans)
                    m, n = 5, 6
                    Am = trans === NoTrans ? (n, m) : (m, n)
                    A = random_matrix(Am[1], Am[2], p, rng)
                    C0 = random_matrix(n, n, p, rng)
                    Cn = BFLA.owned_copy(C0)
                    BFLA.syrk!(Native, triangle, trans, a, A, b, Cn; config = BFLA.KernelConfig(syrk_block = 2))
                    Cb = BFLA.owned_copy(C0)
                    BFLA.syrk!(Native, triangle, trans, a, A, b, Cb)
                    @test all(Cn[i, j] == Cb[i, j] for i in 1:n, j in 1:n)
                end
            end

            @testset "cholesky blocked residual" begin
                A = make_spd(10, p)
                F = BFLA.cholesky(Native, BFLA.owned_copy(A); config = BFLA.KernelConfig(cholesky_block = 4))
                @test issuccess(F)
                L = factor_matrix(F)
                # Reconstruct L·Lᵀ using only the authoritative lower triangle.
                Lc = BFLA.owned_copy(L)
                for j in 1:10, i in 1:(j - 1)
                    Lc[i, j] = BigFloat(0; precision = p)
                end
                Z = BFLA.owned_zeros(BigFloat, 10, 10; precision_bits = p)
                BFLA.gemm!(Native, NoTrans, Trans, one, Lc, Lc, BigFloat(0; precision = p), Z)
                @test Float64(array_difference_norminf(Z, A)) < 1e-20
            end

            @testset "trsm blocked residual" begin
                n, r = 10, 4
                for triangle in (Lower, Upper)
                    T = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
                    for j in 1:n, i in 1:n
                        if triangle === Lower && i < j
                            continue
                        elseif triangle === Upper && i > j
                            continue
                        end
                        T[i, j] = i == j ? BigFloat(n + i; precision = p) : BigFloat(rand(rng); precision = p)
                    end
                    X0 = random_matrix(n, r, p, rng)
                    Xn = BFLA.owned_copy(X0)
                    BFLA.trsm!(Native, LeftSide, triangle, NoTrans, NonUnitDiagonal, one, T, Xn; config = BFLA.KernelConfig(trsm_block = 4))
                    R = BFLA.owned_zeros(BigFloat, n, r; precision_bits = p)
                    BFLA.gemm!(Native, NoTrans, NoTrans, one, T, Xn, BigFloat(0; precision = p), R)
                    @test Float64(array_difference_norminf(R, X0)) < 1e-20
                end
            end
        end
    end
end
