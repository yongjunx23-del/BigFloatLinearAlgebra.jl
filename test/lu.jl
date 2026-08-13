@testset "partial-pivoting LU" begin
    for p in (128, 192, 256, 512)
        @testset "p=$p" begin
            n = 5
            A = random_matrix(n, n, p, MersenneTwister(8000 + p))
            # Force a first-step row swap while retaining a nonsingular matrix.
            A[1, 1] = BigFloat(0; precision = p)
            A[2, 1] = BigFloat(4; precision = p)
            source = BFLA.owned_copy(A)
            F = BFLA.lu(Native, source)
            @test issuccess(F)
            @test factor_kind(F) === :lu
            @test factor_backend(F) === Native
            @test factor_precision(F) == p
            @test source == A
            @test factor_matrix(F) !== source
            @test is_independently_owned(factor_matrix(F))
            @test factor_pivots(F)[1] != 1
            @test sort(factor_perm(F)) == collect(1:n)

            L = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            U = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            for j in 1:n, i in 1:n
                if i > j
                    L[i, j] = factor_matrix(F)[i, j]
                elseif i == j
                    L[i, j] = BigFloat(1; precision = p)
                    U[i, j] = factor_matrix(F)[i, j]
                else
                    U[i, j] = factor_matrix(F)[i, j]
                end
            end
            LU = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p),
                       L, U, BigFloat(0; precision = p), LU)
            PA = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            for j in 1:n, i in 1:n
                PA[i, j] = A[factor_perm(F)[i], j]
            end
            assert_close(LU, PA, p; label = "LU reconstruction")

            xtrue = random_vector(n, p, MersenneTwister(8001 + p))
            b = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
            BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), A, xtrue,
                       BigFloat(0; precision = p), b)
            x = BFLA.solve(F, b)
            assert_close(x, xtrue, p; label = "LU vector solve")
            r = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
            BFLA.residual!(Native, A, x, b, r)
            eta = BFLA.normwise_backward_error(Native, A, x, b, r)
            @test eta <= BigFloat(1000n; precision = p) * eps_bits(p)

            Xtrue = random_matrix(n, 3, p, MersenneTwister(8002 + p))
            B = BFLA.owned_zeros(BigFloat, n, 3; precision_bits = p)
            BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p),
                       A, Xtrue, BigFloat(0; precision = p), B)
            X = BFLA.solve(F, B)
            assert_close(X, Xtrue, p; label = "LU multi-RHS solve")

            Fg = BFLA.lu(Generic, A)
            xg = BFLA.solve(Fg, b)
            assert_close(xg, xtrue, p; label = "Generic LU solve")
            @test factor_backend(Fg) === Generic
            @test factor_diagnostics(F).row_swap_count >= 1
            @test factor_diagnostics(F).failure_position === nothing
        end
    end

    @testset "in-place, singular, and non-finite semantics" begin
        p = 256
        A = random_matrix(4, 4, p, MersenneTwister(8100))
        snapshot = BFLA.owned_copy(A)
        F = BFLA.lu!(Native, A)
        @test factor_matrix(F) === A
        @test A != snapshot

        singular = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        singular[1, 1] = BigFloat(1; precision = p)
        singular[2, 2] = BigFloat(1; precision = p)
        @test_throws LinearAlgebra.SingularException BFLA.lu(
            Native, singular,
        )
        failed = BFLA.lu(Native, singular; check=false)
        @test !issuccess(failed)
        @test factor_status(failed).kind === :singular
        @test factor_status(failed).position == 3
        @test factor_diagnostics(failed).failure_position == 3
        @test BFLA.try_lu!(Native, BFLA.owned_copy(singular)) === nothing
        rhs = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        @test_throws LinearAlgebra.SingularException BFLA.solve!(failed, rhs)

        nonfinite = BFLA.owned_copy(snapshot)
        nonfinite[2, 3] = BigFloat(Inf; precision = p)
        @test_throws DomainError BFLA.lu(Native, nonfinite)
        nanfactor = BFLA.lu(Native, nonfinite; check=false)
        @test factor_status(nanfactor).kind === :nonfinite
        @test BFLA.try_lu!(Native, BFLA.owned_copy(nonfinite)) === nothing
    end

    @testset "small pivots and defensive metadata" begin
        p = 128
        one_by_one = BFLA.owned_zeros(BigFloat, 1, 1; precision_bits = p)
        one_by_one[1, 1] = BigFloat(3; precision = p)
        F1 = BFLA.lu(Native, one_by_one)
        @test size(F1) == (1, 1)
        @test size(F1, 1) == 1
        @test factor_pivots(F1) == [1]
        b1 = BFLA.owned_zeros(BigFloat, 1; precision_bits = p)
        b1[1] = BigFloat(6; precision = p)
        @test BFLA.solve(F1, b1)[1] == BigFloat(2; precision = p)

        two_by_two = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        two_by_two[1, 1] = BigFloat(0; precision = p)
        two_by_two[1, 2] = BigFloat(1; precision = p)
        two_by_two[2, 1] = BigFloat(2; precision = p)
        two_by_two[2, 2] = BigFloat(3; precision = p)
        F2 = BFLA.lu(Native, two_by_two)
        pivots = factor_pivots(F2)
        permutation = factor_perm(F2)
        pivots[1] = 1
        permutation[1] = 1
        @test factor_pivots(F2)[1] == 2
        @test factor_perm(F2) == [2, 1]
        diagnostic_perm = factor_diagnostics(F2).permutation
        diagnostic_perm[1] = 1
        @test factor_diagnostics(F2).permutation == [2, 1]
    end

    @testset "validation and no fallback" begin
        p = 192
        A = random_matrix(4, 4, p, MersenneTwister(8200))
        @test_throws DimensionMismatch BFLA.lu(
            Native, random_matrix(3, 4, p, MersenneTwister(8201)),
        )
        mixed = BFLA.owned_copy(A)
        mixed[4, 4] = BigFloat(mixed[4, 4]; precision = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.lu(Native, mixed)

        F = BFLA.lu(Native, A)
        rhs = random_vector(4, p, MersenneTwister(8202))
        @test_throws ArgumentError BFLA.solve!(F, view(F.factors, :, 1))
        @test_throws DimensionMismatch BFLA.solve!(
            F, BFLA.owned_zeros(BigFloat, 3; precision_bits = p),
        )
        wrong = BFLA.owned_zeros(BigFloat, 4; precision_bits = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.solve!(F, wrong)
        nonfinite_rhs = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        nonfinite_rhs[3] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.solve!(F, nonfinite_rhs)
        finite_rhs = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        F.factors[2, 2] = BigFloat(Inf; precision = p)
        @test_throws DomainError BFLA.solve!(F, finite_rhs)
        F.factors[1, 1] = BigFloat(F.factors[1, 1]; precision = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.solve!(F, wrong)

        struct LUUnsupportedBackend <: BFLA.AbstractBFLABackend end
        unsupported = LUUnsupportedBackend()
        untouched = BFLA.owned_copy(A)
        @test_throws BFLA.UnsupportedOperation BFLA.lu!(unsupported, untouched)
        @test untouched == A
    end

    @testset "capabilities" begin
        @test BFLA.capabilities(Native).lu
        @test BFLA.capabilities(Generic).lu
    end
end
