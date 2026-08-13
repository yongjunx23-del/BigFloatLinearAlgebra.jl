@testset "rank-revealing QR" begin
    for p in (128, 256)
        @testset "p=$p" begin
            @testset "full rank" begin
                A = random_matrix(6, 4, p, MersenneTwister(5000 + p))
                F = BFLA.qr(Native, BFLA.owned_copy(A))
                @test factor_rank(F) == 4
                @test factor_kind(F) === :qr
                # A*P == Q*R reconstruction
                m, n = 6, 4
                Q = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                for i in 1:m
                    Q[i, i] = BigFloat(1; precision = p)
                end
                BFLA.applyQ!(F, Q, NoTrans)
                Rup = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                for j in 1:n, i in 1:j
                    Rup[i, j] = factor_matrix(F)[i, j]
                end
                QR = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), Q, Rup, BigFloat(0; precision = p), QR)
                AP = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                for j in 1:n, i in 1:m
                    AP[i, j] = A[i, factor_jpvt(F)[j]]
                end
                @test Float64(array_difference_norminf(QR, AP)) < 1e-20
            end

            @testset "rank deficiency detected" begin
                A = BFLA.owned_zeros(BigFloat, 4, 3; precision_bits = p)
                A[1, 1] = BigFloat(1; precision = p)
                A[2, 1] = BigFloat(2; precision = p)
                A[3, 1] = BigFloat(3; precision = p)
                A[4, 1] = BigFloat(4; precision = p)
                A[1, 2] = BigFloat(2; precision = p)
                A[2, 2] = BigFloat(4; precision = p)
                A[3, 2] = BigFloat(6; precision = p)
                A[4, 2] = BigFloat(8; precision = p)
                A[1, 3] = BigFloat(1; precision = p)
                A[2, 3] = BigFloat(1; precision = p)
                A[3, 3] = BigFloat(1; precision = p)
                A[4, 3] = BigFloat(1; precision = p)
                tol = BigFloat(1e-10; precision = p)
                F = BFLA.qr(Native, BFLA.owned_copy(A); tol = tol)
                @test factor_rank(F) == 2
                # R[3,3] is below the tolerance
                @test abs(factor_Rdiag(F)[3]) <= tol
            end

            @testset "least-squares solve" begin
                m, n = 8, 5
                A = random_matrix(m, n, p, MersenneTwister(5100 + p))
                x_true = random_vector(n, p, MersenneTwister(5101 + p))
                b = BFLA.owned_zeros(BigFloat, m; precision_bits = p)
                BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), A, x_true, BigFloat(0; precision = p), b)
                Fn = BFLA.qr(Native, BFLA.owned_copy(A))
                Fg = BFLA.qr(Generic, BFLA.owned_copy(A))
                xn = BFLA.owned_copy(b)
                xg = BFLA.owned_copy(b)
                BFLA.solve!(Fn, xn)
                BFLA.solve!(Fg, xg)
                # Both backends solve the same consistent system. Julia's
                # reference QR and Native use distinct legal rounding paths,
                # so compare at an eps-scaled tolerance rather than bitwise.
                @test all(abs(xn[i] - x_true[i]) < BigFloat(1e-10; precision = p) for i in 1:n)
                @test all(abs(xg[i] - x_true[i]) < BigFloat(1e-10; precision = p) for i in 1:n)
                assert_close(view(xn, 1:n), view(xg, 1:n), p; label = "QR solve")
            end

            @testset "Generic packed factor reconstruction" begin
                m, n = 7, 4
                A = random_matrix(m, n, p, MersenneTwister(5150 + p))
                source = BFLA.owned_copy(A)
                F = BFLA.qr!(Generic, source)
                @test factor_matrix(F) === source
                @test is_independently_owned(factor_matrix(F))
                Q = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                for i in 1:m
                    Q[i, i] = BigFloat(1; precision = p)
                end
                BFLA.applyQ!(F, Q)
                R = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                for j in 1:n, i in 1:min(j, m)
                    R[i, j] = factor_matrix(F)[i, j]
                end
                QR = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p),
                           Q, R, BigFloat(0; precision = p), QR)
                AP = BFLA.owned_zeros(BigFloat, m, n; precision_bits = p)
                for j in 1:n, i in 1:m
                    AP[i, j] = A[i, factor_jpvt(F)[j]]
                end
                assert_close(QR, AP, p; label = "Generic QR reconstruction")
            end

            @testset "rank tolerance and column scaling" begin
                A = BFLA.owned_zeros(BigFloat, 5, 3; precision_bits = p)
                one = BigFloat(1; precision = p)
                tiny = BigFloat(
                    BigFloat(2; precision = p)^(-div(p, 3)); precision = p,
                )
                tiny2 = BigFloat(tiny * tiny; precision = p)
                loose_tol = BigFloat(tiny / BigFloat(2; precision = p); precision = p)
                tight_tol = BigFloat(tiny2 / BigFloat(2; precision = p); precision = p)
                A[1, 1] = one
                A[2, 2] = tiny
                A[3, 3] = tiny2
                Floose = BFLA.qr(Native, A; tol = loose_tol)
                @test factor_rank(Floose) == 2
                A2 = BFLA.owned_zeros(BigFloat, 5, 3; precision_bits = p)
                A2[1, 1] = one
                A2[2, 2] = tiny
                A2[3, 3] = tiny2
                Ftight = BFLA.qr(Native, A2; tol = tight_tol)
                @test factor_rank(Ftight) == 3
            end

            @testset "applyQ orthogonal" begin
                m = 6
                A = random_matrix(m, 3, p, MersenneTwister(5200 + p))
                F = BFLA.qr(Native, BFLA.owned_copy(A))
                Q = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                for i in 1:m
                    Q[i, i] = BigFloat(1; precision = p)
                end
                BFLA.applyQ!(F, Q, NoTrans)
                Qt = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                BFLA.gemm!(Native, Trans, NoTrans, BigFloat(1; precision = p), Q, Q, BigFloat(0; precision = p), Qt)
                for i in 1:m, j in 1:m
                    expected = i == j ? BigFloat(1; precision = p) : BigFloat(0; precision = p)
                    @test abs(Qt[i, j] - expected) < BigFloat(1e-20; precision = p)
                end
            end
        end
    end

    @testset "failure and factor invariants" begin
        p = 192
        A = random_matrix(4, 3, p, MersenneTwister(5300))
        @test_throws DomainError BFLA.qr(Native, A; tol = BigFloat(-1; precision = p))
        @test_throws DomainError BFLA.qr(Native, A; tol = BigFloat(Inf; precision = p))
        @test_throws BFLA.PrecisionMismatch BFLA.qr(
            Native, A; tol = BigFloat(0; precision = 128))
        Anan = BFLA.owned_copy(A)
        Anan[2, 2] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.qr(Native, Anan)

        F = BFLA.qr(Native, BFLA.owned_copy(A))
        diagonal = factor_Rdiag(F)
        tolerance = factor_tolerance(F)
        original_diagonal = BigFloat(factor_matrix(F)[1, 1]; precision = p)
        BFLA.MA.operate!(zero, diagonal[1])
        BFLA.MA.operate!(+, tolerance, BigFloat(1; precision = p))
        @test factor_matrix(F)[1, 1] == original_diagonal
        @test factor_tolerance(F) == BigFloat(0; precision = p)
        supplied_tol = BigFloat(1 // 100; precision = p)
        Ftolerance = BFLA.qr(Native, BFLA.owned_copy(A); tol = supplied_tol)
        BFLA.MA.operate!(zero, supplied_tol)
        @test factor_tolerance(Ftolerance) == BigFloat(1 // 100; precision = p)
        @test_throws ArgumentError BFLA.applyQ!(F, view(F.factors, :, 1))
        @test_throws ArgumentError BFLA.solve!(F, view(F.factors, :, 1))
        nonfinite_rhs = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        nonfinite_rhs[2] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.solve!(F, nonfinite_rhs)
        F.tau[1] = BigFloat(Inf; precision = p)
        finite_rhs = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        @test_throws DomainError BFLA.applyQ!(F, finite_rhs)
        F.factors[1, 1] = BigFloat(F.factors[1, 1]; precision = 128)
        rhs = BFLA.owned_zeros(BigFloat, 4; precision_bits = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.solve!(F, rhs)

        wide = random_matrix(3, 4, p, MersenneTwister(5301))
        Fwide = BFLA.qr(Native, wide)
        brhs = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        @test_throws DimensionMismatch BFLA.solve!(Fwide, brhs)
    end
end
