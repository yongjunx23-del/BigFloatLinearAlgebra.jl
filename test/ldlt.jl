@testset "LDLT (Bunch-Kaufman)" begin
    function set_symmetric_owned!(A, row, column, value)
        BFLA.MA.operate_to!(A[row, column], copy, value)
        BFLA.MA.operate_to!(A[column, row], copy, value)
        return A
    end

    function assert_upper_mutation_isolated(F, row, column)
        A = factor_matrix(F)
        @test is_independently_owned(A)
        @test !(A[row, column] === A[column, row])
        authoritative_value = BigFloat(
            A[row, column]; precision = precision(A[row, column]),
        )
        upper = view(A, column:column, row)
        @test !iszero(only(upper))
        BFLA.scal!(
            factor_backend(F),
            BigFloat(2; precision = factor_precision(F)),
            upper,
        )
        @test A[row, column] == authoritative_value
    end

    for p in (128, 256, 512)
        @testset "p=$p" begin
            function saddle(n::Int, seed::Int)
                rng = MersenneTwister(seed)
                m = n ÷ 2
                H = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                for i in 1:m
                    H[i, i] = BigFloat(2; precision = p)
                end
                Ab = BFLA.owned_zeros(BigFloat, m, m; precision_bits = p)
                for j in 1:m, i in 1:m
                    Ab[i, j] = BigFloat(rand(rng); precision = p)
                end
                K = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
                for j in 1:m, i in 1:m
                    BFLA.MA.operate_to!(K[i, j], copy, H[i, j])
                    BFLA.MA.operate_to!(K[i, m + j], copy, Ab[i, j])
                    BFLA.MA.operate_to!(K[m + j, i], copy, Ab[i, j])
                end
                return K
            end

            @testset "saddle-point inertia" begin
                K = saddle(8, 1000 + p)
                F = BFLA.ldlt(Native, BFLA.owned_copy(K))
                @test issuccess(F)
                @test factor_inertia(F) == (4, 4, 0)
                @test factor_kind(F) === :ldlt
                @test factor_diagnostics(F).pivot_1x1_count + 2 * factor_diagnostics(F).pivot_2x2_count == 8
            end

            @testset "2x2 forced pivot" begin
                A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
                A[1, 2] = BigFloat(1; precision = p)
                A[2, 1] = BigFloat(1; precision = p)
                F = BFLA.ldlt(Native, A)
                @test issuccess(F)
                @test factor_inertia(F) == (1, 1, 0)
                @test factor_diagnostics(F).pivot_2x2_count == 1
                diagnostics = factor_diagnostics(F)
                @test diagnostics.min_abs_1x1_pivot === nothing
                @test diagnostics.min_abs_2x2_determinant ==
                      BigFloat(1; precision = p)
                @test diagnostics.min_normalized_2x2_quality ==
                      BigFloat(1; precision = p)
                @test setprecision(BigFloat, 32) do
                    factor_inertia(F) == (1, 1, 0)
                end

                b = random_vector(2, p, MersenneTwister(1500 + p))
                b0 = BFLA.owned_copy(b)
                setprecision(BigFloat, 32) do
                    BFLA.solve!(F, b)
                end
                r = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
                BFLA.residual!(Native, A, b, b0, r)
                @test BFLA.norminf(Native, r) <= BigFloat(10; precision = p) * eps_bits(p)
            end

            @testset "pivot quality diagnostics" begin
                diagonal = BFLA.owned_zeros(
                    BigFloat, 3, 3; precision_bits = p,
                )
                diagonal[1, 1] = BigFloat(-3; precision = p)
                diagonal[2, 2] = BigFloat(1 // 8; precision = p)
                diagonal[3, 3] = BigFloat(5; precision = p)
                F = BFLA.ldlt(Native, diagonal)
                diagnostics = factor_diagnostics(F)
                @test diagnostics.factor_kind === :ldlt
                @test diagnostics.inertia == (2, 1, 0)
                @test diagnostics.min_abs_1x1_pivot ==
                      BigFloat(1 // 8; precision = p)
                @test diagnostics.min_abs_2x2_determinant === nothing
                @test diagnostics.min_normalized_2x2_quality === nothing
                BFLA.MA.operate!(zero, diagnostics.min_abs_1x1_pivot)
                @test factor_diagnostics(F).min_abs_1x1_pivot ==
                      BigFloat(1 // 8; precision = p)

                nearly_singular = BFLA.owned_zeros(
                    BigFloat, 2, 2; precision_bits = p,
                )
                tiny = BigFloat(0; precision = p)
                BFLA._mpfr_set_ui_2exp!(tiny, 1, -div(p, 2))
                nearly_singular[1, 1] = BigFloat(0; precision = p)
                nearly_singular[2, 1] = BigFloat(1; precision = p)
                nearly_singular[2, 2] = tiny
                near_factor = BFLA.ldlt(Native, nearly_singular)
                near_diagnostics = factor_diagnostics(near_factor)
                @test near_diagnostics.min_abs_2x2_determinant ==
                      BigFloat(1; precision = p)
                @test near_diagnostics.min_normalized_2x2_quality ==
                      BigFloat(1; precision = p)

                # Exercise the normalized determinant definition directly on
                # a valid packed 2x2 D block near cancellation. Standard BK
                # pivoting normally avoids selecting such a weak block.
                packed = BFLA.owned_zeros(
                    BigFloat, 2, 2; precision_bits = p,
                )
                packed[1, 1] = BigFloat(1; precision = p)
                packed[2, 1] = BigFloat(1; precision = p)
                BFLA.MA.operate_to!(packed[2, 2], +,
                    BigFloat(1; precision = p), tiny)
                packed_factor = BFLA.BFLALDLTFactor(
                    packed,
                    Native,
                    p,
                    BFLA.FactorStatus(:success, nothing),
                    [1, 2],
                    [2],
                    BitVector((false, true)),
                )
                packed_diagnostics = factor_diagnostics(packed_factor)
                @test packed_diagnostics.min_abs_2x2_determinant == tiny
                @test packed_diagnostics.min_normalized_2x2_quality > 0
                @test packed_diagnostics.min_normalized_2x2_quality < tiny
            end

            @testset "factorization ignores ambient precision" begin
                # Put the leading diagonal one p-bit ulp below the exact BK
                # threshold alpha. A low ambient context must not round pivot
                # comparisons or scratch away from factor precision.
                alpha = setprecision(BigFloat, p) do
                    (BigFloat(1; precision = p) +
                     sqrt(BigFloat(17; precision = p))) /
                    BigFloat(8; precision = p)
                end
                A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
                A[1, 1] = prevfloat(alpha)
                A[2, 1] = BigFloat(1; precision = p)
                A[1, 2] = BigFloat(1; precision = p)
                A[2, 2] = BigFloat(2; precision = p)
                A[3, 3] = BigFloat(3; precision = p)

                baseline = BFLA.ldlt(Native, BFLA.owned_copy(A))
                low_ambient = setprecision(BigFloat, 32) do
                    BFLA.ldlt(Native, BFLA.owned_copy(A))
                end
                @test factor_perm(low_ambient) == factor_perm(baseline)
                @test factor_blocks(low_ambient) == factor_blocks(baseline)
                @test all(
                    factor_matrix(low_ambient)[i, j] ==
                    factor_matrix(baseline)[i, j]
                    for j in 1:3 for i in j:3
                )
            end

            @testset "solve residual" begin
                n = 8
                K = saddle(n, 2000 + p)
                b = random_vector(n, p, MersenneTwister(2000 + p))
                b0 = BFLA.owned_copy(b)
                Fn = BFLA.ldlt(Native, BFLA.owned_copy(K))
                Fg = BFLA.ldlt(Generic, BFLA.owned_copy(K))
                @test issuccess(Fn) && issuccess(Fg)
                xn = BFLA.owned_copy(b)
                xg = BFLA.owned_copy(b)
                BFLA.solve!(Fn, xn)
                BFLA.solve!(Fg, xg)
                assert_close(xn, xg, p; label = "ldlt solve")
                r = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
                BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), K, xn, BigFloat(0; precision = p), r)
                for i in 1:n
                    r[i] = r[i] - b0[i]
                end
                @test Float64(BFLA.norminf(Native, r)) < 1e-20
            end

            @testset "multi-RHS solve" begin
                n = 8
                K = saddle(n, 3000 + p)
                rhs = random_matrix(n, 3, p, MersenneTwister(3000 + p))
                rhs0 = BFLA.owned_copy(rhs)
                F = BFLA.ldlt(Native, BFLA.owned_copy(K))
                BFLA.solve!(F, rhs)
                R = BFLA.owned_zeros(BigFloat, n, 3; precision_bits = p)
                BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), K, rhs, BigFloat(0; precision = p), R)
                for i in eachindex(R)
                    R[i] = R[i] - rhs0[i]
                end
                @test Float64(BFLA.norminf(Native, R)) < 1e-20
            end

            @testset "singular fails closed" begin
                A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
                A[1, 1] = BigFloat(1; precision = p)
                F = BFLA.ldlt!(Native, A; check = false)
                @test !issuccess(F)
                @test factor_status(F).kind === :pivot_failure
                @test factor_diagnostics(F).inertia === nothing
                @test_throws LinearAlgebra.SingularException BFLA.ldlt!(Native, BFLA.owned_copy(A))
                @test BFLA.try_ldlt!(Native, BFLA.owned_copy(A)) === nothing
            end

            @testset "authoritative lower triangle and non-finite input" begin
                clean = saddle(6, 3500 + p)
                poisoned = BFLA.owned_copy(clean)
                for j in 2:6, i in 1:(j - 1)
                    poisoned[i, j] = BigFloat(NaN; precision = p)
                end
                Fclean = BFLA.ldlt(Native, clean)
                Fpoison = BFLA.ldlt(Native, poisoned)
                @test issuccess(Fpoison)
                @test factor_perm(Fpoison) == factor_perm(Fclean)
                @test factor_blocks(Fpoison) == factor_blocks(Fclean)
                for j in 1:6, i in j:6
                    @test factor_matrix(Fpoison)[i, j] == factor_matrix(Fclean)[i, j]
                end

                lower_nan = BFLA.owned_copy(clean)
                lower_nan[4, 2] = BigFloat(NaN; precision = p)
                @test_throws DomainError BFLA.ldlt(Native, lower_nan)
                failed = BFLA.ldlt(Native, lower_nan; check = false)
                @test factor_status(failed).kind === :nonfinite
                @test factor_diagnostics(failed).inertia === nothing
                @test BFLA.try_ldlt!(
                    Native, BFLA.owned_copy(lower_nan),
                ) === nothing
            end

            @testset "2x2 pivots with permutation and solve" begin
                # Deterministic symmetric indefinite matrix with two 2×2
                # pivots, one of which requires a non-trivial symmetric
                # permutation (the leading [[0,1],[1,0]] block swaps rows 2,3).
                n = 5
                A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
                A[1, 3] = BigFloat(1; precision = p)
                A[3, 1] = BigFloat(1; precision = p)
                A[2, 2] = BigFloat(1; precision = p)
                A[4, 5] = BigFloat(1; precision = p)
                A[5, 4] = BigFloat(1; precision = p)
                A[3, 3] = BigFloat(2; precision = p)
                b = random_vector(n, p, MersenneTwister(4000 + p))
                b0 = BFLA.owned_copy(b)
                F = BFLA.ldlt(Native, BFLA.owned_copy(A))
                @test issuccess(F)
                @test factor_diagnostics(F).pivot_2x2_count >= 1
                @test factor_perm(F) != collect(1:n)  # permutation is non-trivial
                BFLA.solve!(F, b)
                r = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
                BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), A, b, BigFloat(0; precision = p), r)
                for i in 1:n
                    r[i] = r[i] - b0[i]
                end
                @test Float64(BFLA.norminf(Native, r)) < 1e-15
            end
        end
    end


    @testset "solve non-finite and alias failures" begin
        p = 192
        A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        A[1, 2] = BigFloat(1; precision = p)
        A[2, 1] = BigFloat(1; precision = p)
        F = BFLA.ldlt(Native, A)
        @test_throws ArgumentError BFLA.solve!(F, view(F.factors, :, 1))
        rhs = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        rhs[1] = BigFloat(NaN; precision = p)
        @test_throws DomainError BFLA.solve!(F, rhs)
        F.factors[2, 1] = BigFloat(Inf; precision = p)
        finite_rhs = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
        @test_throws DomainError BFLA.solve!(F, finite_rhs)
        @test_throws DomainError factor_inertia(F)
    end

    @testset "factor storage remains independently owned" begin
        p = 192

        one_by_one = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        one_by_one[1, 1] = BigFloat(4; precision = p)
        one_by_one[2, 2] = BigFloat(3; precision = p)
        one_by_one[3, 3] = BigFloat(2; precision = p)
        set_symmetric_owned!(one_by_one, 2, 1, BigFloat(1; precision = p))
        set_symmetric_owned!(one_by_one, 3, 2, BigFloat(1; precision = p))

        native_1x1 = BFLA.ldlt(Native, one_by_one)
        @test factor_blocks(native_1x1) == [1, 1, 1]
        assert_upper_mutation_isolated(native_1x1, 2, 1)

        generic_1x1 = BFLA.ldlt(Generic, one_by_one)
        @test factor_blocks(generic_1x1) == [1, 1, 1]
        assert_upper_mutation_isolated(generic_1x1, 2, 1)

        forced_2x2 = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        set_symmetric_owned!(forced_2x2, 2, 1, BigFloat(1; precision = p))
        set_symmetric_owned!(forced_2x2, 3, 1, BigFloat(1 // 4; precision = p))
        set_symmetric_owned!(forced_2x2, 3, 2, BigFloat(1 // 3; precision = p))
        forced_2x2[3, 3] = BigFloat(2; precision = p)
        forced_2x2[4, 4] = BigFloat(3; precision = p)

        native_2x2 = BFLA.ldlt(Native, forced_2x2)
        @test first(factor_blocks(native_2x2)) == 2
        @test length(factor_blocks(native_2x2)) >= 2
        assert_upper_mutation_isolated(native_2x2, 2, 1)
        assert_upper_mutation_isolated(native_2x2, 3, 1)

        generic_2x2 = BFLA.ldlt(Generic, forced_2x2)
        @test first(factor_blocks(generic_2x2)) == 2
        assert_upper_mutation_isolated(generic_2x2, 2, 1)
        assert_upper_mutation_isolated(generic_2x2, 3, 1)

        permuted = BFLA.owned_zeros(BigFloat, 5, 5; precision_bits = p)
        set_symmetric_owned!(permuted, 3, 1, BigFloat(1; precision = p))
        permuted[2, 2] = BigFloat(1; precision = p)
        permuted[3, 3] = BigFloat(2; precision = p)
        set_symmetric_owned!(permuted, 5, 4, BigFloat(1; precision = p))

        for backend in (Native, Generic)
            F = BFLA.ldlt(backend, permuted)
            @test factor_perm(F) != collect(1:5)
            @test is_independently_owned(factor_matrix(F))
            assert_upper_mutation_isolated(F, 3, 1)
        end
    end

    @testset "in-place input ownership fails closed" begin
        p = 192
        shared = BigFloat(1; precision = p)
        A = fill(shared, 2, 2)
        snapshot = copy(A)
        identities = [A[index] for index in eachindex(A)]

        for backend in (Native, Generic)
            target = copy(A)
            @test_throws ArgumentError BFLA.ldlt!(backend, target)
            @test all(
                target[index] == snapshot[index] for index in eachindex(target)
            )
            @test all(
                target[index] === identities[position]
                for (position, index) in enumerate(eachindex(target))
            )
        end

        # The allocating API explicitly breaks source sharing before invoking
        # the in-place factorization.
        source = fill(BigFloat(0; precision = p), 3, 3)
        source[1, 1] = BigFloat(2; precision = p)
        source[2, 2] = BigFloat(3; precision = p)
        source[3, 3] = BigFloat(4; precision = p)
        for backend in (Native, Generic)
            F = BFLA.ldlt(backend, source)
            @test issuccess(F)
            @test is_independently_owned(factor_matrix(F))
        end

        upper_shared = BigFloat(NaN; precision = p)
        lower_authoritative = BFLA.owned_zeros(
            BigFloat, 3, 3; precision_bits = p,
        )
        for index in 1:3
            lower_authoritative[index, index] = BigFloat(
                index + 1; precision = p,
            )
        end
        lower_authoritative[1, 2] = upper_shared
        lower_authoritative[1, 3] = upper_shared
        lower_authoritative[2, 3] = upper_shared
        for backend in (Native, Generic)
            target = BFLA.owned_copy(lower_authoritative)
            target[1, 2] = upper_shared
            target[1, 3] = upper_shared
            target[2, 3] = upper_shared
            F_upper_shared = BFLA.ldlt!(backend, target)
            @test issuccess(F_upper_shared)
            @test is_independently_owned(factor_matrix(F_upper_shared))
        end
    end
end
