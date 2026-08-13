@testset "residual and backward error" begin
    for p in (128, 192, 256, 512)
        @testset "p=$p" begin
            rng = MersenneTwister(7000 + p)
            m, n = 7, 4
            A = random_matrix(m, n, p, rng)
            x = random_vector(n, p, rng)
            b = random_vector(m, p, rng)
            r = BFLA.owned_zeros(BigFloat, m; precision_bits = p)
            BFLA.residual!(Native, A, x, b, r)

            reference = BFLA.owned_copy(b)
            BFLA.gemv!(Generic, NoTrans, BigFloat(-1; precision = p), A, x,
                       BigFloat(1; precision = p), reference)
            assert_close(r, reference, p; label = "vector residual")
            @test is_independently_owned(r)
            @test all(precision(value) == p for value in r)

            eta = BFLA.normwise_backward_error(Native, A, x, b, r)
            eta_alloc = BFLA.normwise_backward_error(Native, A, x, b)
            @test eta == eta_alloc
            @test precision(eta) == p
            @test eta >= 0

            # Transposed multi-RHS path.
            nrhs = 3
            X = random_matrix(m, nrhs, p, rng)
            B = random_matrix(n, nrhs, p, rng)
            R = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits = p)
            Rg = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits = p)
            BFLA.residual!(Native, Trans, A, X, B, R)
            BFLA.residual!(Generic, Trans, A, X, B, Rg)
            assert_close(R, Rg, p; label = "multi-RHS transposed residual")
            eta_matrix = BFLA.normwise_backward_error(Native, Trans, A, X, B, R)
            @test isfinite(eta_matrix)
            @test eta_matrix >= 0
        end
    end

    @testset "known values and zero denominator" begin
        p = 256
        A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        A[1, 1] = BigFloat(2; precision = p)
        A[2, 2] = BigFloat(4; precision = p)
        x = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        x[1] = BigFloat(1; precision = p)
        x[2] = BigFloat(2; precision = p)
        b = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        b[1] = BigFloat(3; precision = p)
        b[2] = BigFloat(7; precision = p)
        r = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        BFLA.residual!(Native, A, x, b, r)
        @test r == [BigFloat(1; precision = p), BigFloat(-1; precision = p)]
        # ||r||inf / (||A||inf*||x||inf + ||b||inf) = 1/(4*2+7)
        expected = BigFloat(1; precision = p) / BigFloat(15; precision = p)
        @test BFLA.normwise_backward_error(Native, A, x, b, r) == expected

        zero_A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        zero_x = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        zero_b = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        zero_r = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        @test iszero(BFLA.normwise_backward_error(
            Native, zero_A, zero_x, zero_b, zero_r,
        ))
        zero_r[1] = BigFloat(1; precision = p)
        @test isinf(BFLA.normwise_backward_error(
            Native, zero_A, zero_x, zero_b, zero_r,
        ))
    end

    @testset "failure semantics" begin
        p = 256
        A = random_matrix(3, 2, p, MersenneTwister(7100))
        x = random_vector(2, p, MersenneTwister(7101))
        b = random_vector(3, p, MersenneTwister(7102))
        r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        @test_throws ArgumentError BFLA.residual!(Native, A, x, b, b)
        @test_throws DimensionMismatch BFLA.residual!(Native, A, x, b[1:2], r)
        @test_throws DimensionMismatch BFLA.residual!(
            Native, A, reshape(x, 2, 1), b, r,
        )

        mixed = BFLA.owned_copy(r)
        mixed[2] = BigFloat(0; precision = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.residual!(
            Native, A, x, b, mixed,
        )
        Anan = BFLA.owned_copy(A)
        Anan[1, 1] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.residual!(Native, Anan, x, b, r)
        rnan = BFLA.owned_copy(r)
        rnan[1] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.normwise_backward_error(
            Native, A, x, b, rnan,
        )

        struct QualityUnsupportedBackend <: BFLA.AbstractBFLABackend end
        unsupported = QualityUnsupportedBackend()
        snapshot = BFLA.owned_copy(r)
        @test_throws BFLA.UnsupportedOperation BFLA.residual!(
            unsupported, A, x, b, r,
        )
        @test r == snapshot
        @test_throws BFLA.UnsupportedOperation BFLA.normwise_backward_error(
            unsupported, A, x, b, r,
        )
    end

    @testset "capabilities" begin
        for backend in (Native, Generic)
            caps = BFLA.capabilities(backend)
            @test caps.multi_rhs
            @test caps.residual
            @test caps.backward_error
            @test caps.higher_precision_residual
            @test caps.refinement
        end
    end

    @testset "explicit higher-precision residual" begin
        for (p, q) in ((128, 256), (192, 384), (256, 512))
            rng = MersenneTwister(7200 + p)
            m, n = 6, 4
            A = random_matrix(m, n, p, rng)
            x = random_vector(n, p, rng)
            b = random_vector(m, p, rng)
            A0 = BFLA.owned_copy(A)
            x0 = BFLA.owned_copy(x)
            b0 = BFLA.owned_copy(b)
            r = BFLA.owned_zeros(BigFloat, m; precision_bits = q)

            report = setprecision(BigFloat, 32) do
                BFLA.higher_precision_residual!(
                    Native, A, x, b, r; residual_precision = q,
                )
            end
            @test report.residual === r
            @test report.factor_precision == p
            @test report.residual_precision == q
            @test precision(report.backward_error) == q
            @test all(precision(value) == q for value in r)
            @test is_independently_owned(r)
            @test A == A0 && x == x0 && b == b0

            Aq = BFLA.owned_copy(A; precision_bits = q)
            xq = BFLA.owned_copy(x; precision_bits = q)
            bq = BFLA.owned_copy(b; precision_bits = q)
            reference = BFLA.owned_zeros(BigFloat, m; precision_bits = q)
            BFLA.residual!(Generic, Aq, xq, bq, reference)
            assert_close(r, reference, q; label = "higher precision residual")
            expected_error = BFLA.normwise_backward_error(
                Generic, Aq, xq, bq, reference,
            )
            @test report.backward_error == expected_error

            rg = BFLA.owned_zeros(BigFloat, m; precision_bits = q)
            greport = BFLA.higher_precision_residual!(
                Generic, A, x, b, rg; residual_precision = q,
            )
            assert_close(rg, reference, q; label = "Generic high residual")
            @test greport.factor_precision == p
            @test greport.residual_precision == q
        end

        p, q = 128, 256
        A = random_matrix(4, 3, p, MersenneTwister(7300))
        X = random_matrix(4, 2, p, MersenneTwister(7301))
        B = random_matrix(3, 2, p, MersenneTwister(7302))
        R = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = q)
        report = BFLA.higher_precision_residual!(
            Native, Trans, A, X, B, R; residual_precision = q,
        )
        Aq = BFLA.owned_copy(A; precision_bits = q)
        Xq = BFLA.owned_copy(X; precision_bits = q)
        Bq = BFLA.owned_copy(B; precision_bits = q)
        Rq = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits = q)
        BFLA.residual!(Generic, Trans, Aq, Xq, Bq, Rq)
        assert_close(R, Rq, q; label = "high precision transposed multi-RHS")
        @test report.residual_precision == q
    end

    @testset "higher-precision failure semantics" begin
        p = 256
        A = random_matrix(3, 2, p, MersenneTwister(7400))
        x = random_vector(2, p, MersenneTwister(7401))
        b = random_vector(3, p, MersenneTwister(7402))
        same = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        @test_throws ArgumentError BFLA.higher_precision_residual!(
            Native, A, x, b, same; residual_precision = p,
        )
        lower = BFLA.owned_zeros(BigFloat, 3; precision_bits = 128)
        @test_throws ArgumentError BFLA.higher_precision_residual!(
            Native, A, x, b, lower; residual_precision = 128,
        )
        high = BFLA.owned_zeros(BigFloat, 3; precision_bits = 512)
        @test_throws BFLA.PrecisionMismatch BFLA.higher_precision_residual!(
            Native, A, x, b, high;
            residual_precision = 512,
            factor_precision = 128,
        )
        @test_throws BFLA.PrecisionMismatch BFLA.higher_precision_residual!(
            Native, A, x, b, high; residual_precision = 384,
        )
        mixed = BFLA.owned_copy(high)
        mixed[2] = BigFloat(0; precision = 384)
        @test_throws BFLA.PrecisionMismatch BFLA.higher_precision_residual!(
            Native, A, x, b, mixed; residual_precision = 512,
        )
        @test_throws ArgumentError BFLA.higher_precision_residual!(
            Native, A, x, b, view(A, :, 1); residual_precision = p,
        )

        unsupported = QualityUnsupportedBackend()
        snapshot = BFLA.owned_copy(high)
        @test_throws BFLA.UnsupportedOperation BFLA.higher_precision_residual!(
            unsupported, A, x, b, high; residual_precision = 512,
        )
        @test high == snapshot
    end

    @testset "one-step refinement" begin
        for p in (128, 256)
            q = 2p
            n = 6
            A = make_spd(n, p; seed = 7500 + p, delta_bits = div(p, 3))
            x_true = random_vector(n, p, MersenneTwister(7501 + p))
            b = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
            BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), A, x_true,
                       BigFloat(0; precision = p), b)
            F = BFLA.cholesky(Native, A)
            x = BFLA.solve(F, b)
            residual = BFLA.owned_zeros(BigFloat, n; precision_bits = q)
            correction = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
            A0 = BFLA.owned_copy(A)
            b0 = BFLA.owned_copy(b)
            factors0 = BFLA.owned_copy(factor_matrix(F))

            report = BFLA.refine_once!(F, A, x, b, residual, correction)
            @test report.x === x
            @test report.residual === residual
            @test report.correction === correction
            @test report.backend === Native
            @test report.factor_precision == p
            @test report.residual_precision == q
            @test precision(report.backward_error_before) == q
            @test precision(report.backward_error_after) == q
            @test report.backward_error_after <= report.backward_error_before
            @test A == A0 && b == b0
            @test factor_matrix(F) == factors0
            @test all(precision(value) == p for value in correction)
            @test all(precision(value) == q for value in residual)
            @test is_independently_owned(correction)
            @test is_independently_owned(residual)
        end

        @testset "authoritative factor storage" begin
            p, q = 128, 256
            A = make_spd(4, p; seed = 7550)
            b = random_vector(4, p, MersenneTwister(7551))
            F = BFLA.cholesky(Native, A)
            # Lower Cholesky never reads the packed factor's upper triangle.
            for j in 2:4, i in 1:(j - 1)
                factor_matrix(F)[i, j] = BigFloat(NaN; precision = p)
            end
            x = BFLA.solve(F, b)
            r = BFLA.owned_zeros(BigFloat, 4; precision_bits = q)
            d = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
            @test_nowarn BFLA.refine_once!(F, A, x, b, r, d)
            @test all(isfinite, x)
        end

        @testset "same-precision multi-RHS LDLT" begin
            p = 192
            n, nrhs = 4, 2
            A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            for i in 1:n
                A[i, i] = BigFloat(isodd(i) ? i + 2 : -(i + 2); precision = p)
            end
            Xtrue = random_matrix(n, nrhs, p, MersenneTwister(7600))
            B = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits = p)
            BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p),
                       A, Xtrue, BigFloat(0; precision = p), B)
            F = BFLA.ldlt(Native, A)
            X = BFLA.solve(F, B)
            # Perturb the initial solution so a nonzero correction is required.
            X[1, 1] = BigFloat(X[1, 1] + BigFloat(2; precision = p)^(-80);
                               precision = p)
            R = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits = p)
            D = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits = p)
            report = BFLA.refine_once!(F, A, X, B, R, D)
            @test report.residual_precision == p
            @test report.backward_error_after <= report.backward_error_before
            assert_close(X, Xtrue, p; label = "multi-RHS refinement")
        end
    end

    @testset "refinement failure semantics" begin
        p, q = 128, 256
        A = make_spd(3, p; seed = 7700)
        F = BFLA.cholesky(Native, A)
        x = random_vector(3, p, MersenneTwister(7701))
        b = random_vector(3, p, MersenneTwister(7702))
        r = BFLA.owned_zeros(BigFloat, 3; precision_bits = q)
        d = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        x0 = BFLA.owned_copy(x)
        r0 = BFLA.owned_copy(r)
        d0 = BFLA.owned_copy(d)

        @test_throws ArgumentError BFLA.refine_once!(F, A, x, b, r, x)
        @test x == x0 && r == r0 && d == d0
        low_r = BFLA.owned_zeros(BigFloat, 3; precision_bits = 64)
        @test_throws ArgumentError BFLA.refine_once!(F, A, x, b, low_r, d)
        wrong_d = BFLA.owned_zeros(BigFloat, 3; precision_bits = q)
        @test_throws BFLA.PrecisionMismatch BFLA.refine_once!(
            F, A, x, b, r, wrong_d,
        )
        @test_throws ArgumentError BFLA.refine_once!(
            F, factor_matrix(F), x, b, r, d,
        )
        @test_throws BFLA.UnsupportedOperation BFLA.refine_once!(
            F, A, x, b, r, d; trans = Trans,
        )

        bad_factor = BFLA.cholesky!(Native, begin
            M = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            M[1, 1] = BigFloat(1; precision = p)
            M[2, 2] = BigFloat(-1; precision = p)
            M
        end; check = false)
        badx = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        badb = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        badr = BFLA.owned_zeros(BigFloat, 2; precision_bits = q)
        badd = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        @test_throws ArgumentError BFLA.refine_once!(
            bad_factor, A[1:2, 1:2], badx, badb, badr, badd,
        )

        qr_factor = BFLA.qr(Native, A)
        qr_x = BFLA.solve(qr_factor, b)
        qr_r = BFLA.owned_zeros(BigFloat, 3; precision_bits = q)
        qr_d = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        qr_x0 = BFLA.owned_copy(qr_x)
        qr_r0 = BFLA.owned_copy(qr_r)
        qr_d0 = BFLA.owned_copy(qr_d)
        BFLA.MA.operate_to!(
            qr_factor.effective_threshold,
            copy,
            BigFloat(NaN; precision = p),
        )
        @test_throws DomainError BFLA.refine_once!(
            qr_factor, A, qr_x, b, qr_r, qr_d,
        )
        @test qr_x == qr_x0 && qr_r == qr_r0 && qr_d == qr_d0

        clean_qr = BFLA.qr(Native, A)
        mixed_qr = BFLA.BFLAQRFactor(
            factor_matrix(clean_qr),
            factor_backend(clean_qr),
            factor_precision(clean_qr),
            factor_status(clean_qr),
            clean_qr.tau,
            factor_jpvt(clean_qr),
            factor_rank(clean_qr),
            factor_tolerance(clean_qr),
            factor_rank_atol(clean_qr),
            factor_rank_rtol(clean_qr),
            BigFloat(factor_rank_scale(clean_qr); precision = 64),
            factor_rank_threshold(clean_qr),
        )
        mixed_x = BFLA.solve(clean_qr, b)
        mixed_r = BFLA.owned_zeros(BigFloat, 3; precision_bits = q)
        mixed_d = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        mixed_x0 = BFLA.owned_copy(mixed_x)
        @test_throws BFLA.PrecisionMismatch BFLA.refine_once!(
            mixed_qr, A, mixed_x, b, mixed_r, mixed_d,
        )
        @test mixed_x == mixed_x0
        @test all(iszero, mixed_r) && all(iszero, mixed_d)
    end

    @testset "refinement capability" begin
        @test BFLA.capabilities(Native).refinement
        @test BFLA.capabilities(Generic).refinement
    end
end
