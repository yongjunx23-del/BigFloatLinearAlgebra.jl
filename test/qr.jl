@testset "rank-revealing QR" begin
    for p in (128, 256)
        @testset "p=$p" begin
            @testset "full rank" begin
                A = random_matrix(6, 4, p, MersenneTwister(5000 + p))
                F = BFLA.qr(Native, BFLA.owned_copy(A))
                @test factor_rank(F) == 4
                @test factor_kind(F) === :rrqr
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

            @testset "relative-rank scale invariance" begin
                binary_power(exponent) = begin
                    value = BigFloat(0; precision = p)
                    BFLA._mpfr_set_ui_2exp!(value, 1, exponent)
                    value
                end
                base = BFLA.owned_zeros(BigFloat, 5, 3; precision_bits = p)
                base[1, 2] = BigFloat(1; precision = p)
                base[2, 1] = binary_power(-20)
                base[3, 3] = binary_power(-80)
                atol = BigFloat(0; precision = p)
                rtol = binary_power(-60)
                tighter_rtol = binary_power(-90)

                for backend in (Native, Generic), exponent in (-200, -100, 0, 100, 200)
                    scale = binary_power(exponent)
                    A = BFLA.owned_zeros(BigFloat, size(base)...; precision_bits = p)
                    for index in eachindex(A, base)
                        BFLA.MA.operate_to!(A[index], *, base[index], scale)
                    end
                    F = BFLA.qr(backend, A; atol = atol, rtol = rtol)
                    @test factor_kind(F) === :rrqr
                    @test factor_triangle(F) === nothing
                    @test factor_rank(F) == 2
                    @test numerical_rank(F) == 2
                    @test numerical_rank(F; atol = atol, rtol = tighter_rtol) == 3
                    @test factor_jpvt(F)[1] == 2
                    @test factor_rank_atol(F) == atol
                    @test factor_rank_rtol(F) == rtol
                    @test factor_rank_scale(F) == abs(scale)
                    @test factor_rank_threshold(F) == rtol * abs(scale)
                    diagnostics = factor_diagnostics(F)
                    @test diagnostics.rank == 2
                    @test diagnostics.reference_scale == abs(scale)
                    @test diagnostics.effective_threshold == rtol * abs(scale)
                    @test diagnostics.failure_position === nothing
                    @test diagnostics.min_accepted_abs_Rdiag >
                          diagnostics.effective_threshold
                    @test diagnostics.next_rejected_abs_Rdiag <=
                          diagnostics.effective_threshold
                end

                exact = BFLA.owned_copy(base)
                exact[3, 3] = BigFloat(0; precision = p)
                for backend in (Native, Generic)
                    F = BFLA.qr(backend, exact; atol = atol, rtol = rtol)
                    @test factor_rank(F) == 2
                    @test numerical_rank(F; atol = atol, rtol = tighter_rtol) == 2
                end
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
        @test_throws DomainError BFLA.qr(
            Native, A; rtol = BigFloat(-1; precision = p),
        )
        @test_throws DomainError BFLA.qr(
            Native, A; atol = BigFloat(Inf; precision = p),
        )
        @test_throws ArgumentError BFLA.qr(
            Native,
            A;
            tol = BigFloat(0; precision = p),
            rtol = BigFloat(0; precision = p),
        )
        @test_throws BFLA.PrecisionMismatch BFLA.qr(
            Native, A; tol = BigFloat(0; precision = 128))
        @test_throws BFLA.PrecisionMismatch BFLA.qr(
            Native, A; rtol = BigFloat(0; precision = 128),
        )
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
        @test factor_rank_atol(F) == BigFloat(0; precision = p)
        @test factor_rank_rtol(F) ==
              BigFloat(max(size(A)...); precision = p) * eps_bits(p)
        @test factor_rank_threshold(F) ==
              factor_rank_rtol(F) * factor_rank_scale(F)
        supplied_tol = BigFloat(1 // 100; precision = p)
        Ftolerance = BFLA.qr(Native, BFLA.owned_copy(A); tol = supplied_tol)
        BFLA.MA.operate!(zero, supplied_tol)
        @test factor_tolerance(Ftolerance) == BigFloat(1 // 100; precision = p)
        @test factor_rank_atol(Ftolerance) == BigFloat(1 // 100; precision = p)
        @test iszero(factor_rank_rtol(Ftolerance))

        metadata = factor_diagnostics(Ftolerance)
        BFLA.MA.operate!(zero, metadata.R_diagonal[1])
        metadata.permutation[1] = 0
        BFLA.MA.operate!(zero, metadata.effective_threshold)
        @test factor_Rdiag(Ftolerance)[1] != 0
        @test all(>(0), factor_jpvt(Ftolerance))
        @test factor_rank_threshold(Ftolerance) == BigFloat(1 // 100; precision = p)

        rank_one = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = p)
        rank_one[1, 1] = BigFloat(4; precision = p)
        rank_one[2, 2] = BigFloat(1 // 1000; precision = p)
        Frankone = BFLA.qr(
            Native,
            rank_one;
            atol = BigFloat(1 // 100; precision = p),
            rtol = BigFloat(0; precision = p),
        )
        rank_diagnostics = factor_diagnostics(Frankone)
        @test rank_diagnostics.rank == 1
        @test rank_diagnostics.min_accepted_abs_Rdiag ==
              abs(rank_diagnostics.R_diagonal[1])
        @test rank_diagnostics.next_rejected_abs_Rdiag ==
              abs(rank_diagnostics.R_diagonal[2])

        metadata_source = BFLA.qr(Native, BFLA.owned_copy(A))
        metadata_mixed = BFLA.BFLAQRFactor(
            factor_matrix(metadata_source),
            factor_backend(metadata_source),
            factor_precision(metadata_source),
            factor_status(metadata_source),
            metadata_source.tau,
            factor_jpvt(metadata_source),
            factor_rank(metadata_source),
            factor_tolerance(metadata_source),
            factor_rank_atol(metadata_source),
            factor_rank_rtol(metadata_source),
            BigFloat(factor_rank_scale(metadata_source); precision = 128),
            factor_rank_threshold(metadata_source),
        )
        @test_throws BFLA.PrecisionMismatch factor_diagnostics(metadata_mixed)

        metadata_nonfinite = BFLA.qr(Native, BFLA.owned_copy(A))
        BFLA.MA.operate_to!(
            metadata_nonfinite.effective_threshold,
            copy,
            BigFloat(NaN; precision = p),
        )
        @test_throws DomainError factor_diagnostics(metadata_nonfinite)
        @test_throws BFLA.PrecisionMismatch BFLA.numerical_rank(
            F; atol = BigFloat(0; precision = 128),
        )
        @test_throws DomainError BFLA.numerical_rank(
            F; rtol = BigFloat(NaN; precision = p),
        )
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
