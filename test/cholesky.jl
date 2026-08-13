function solve_residual(S::AbstractMatrix{BigFloat}, x::AbstractVecOrMat{BigFloat}, b::AbstractVecOrMat{BigFloat})
    p = precision(first(S))
    R = BFLA.owned_zeros(BigFloat, size(b)...; precision_bits = p)
    if x isa AbstractVector
        BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), S, x, BigFloat(0; precision = p), vec(R))
    else
        BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), S, x, BigFloat(0; precision = p), R)
    end
    setprecision(BigFloat, p) do
        for i in eachindex(R)
            R[i] = R[i] - b[i]
        end
    end
    return BFLA.norminf(Native, R)
end

function legacy_style_cholesky_lower!(A, p)
    n = size(A, 1)
    accumulator = BigFloat(0; precision = p)
    multiplication_buffer = BigFloat(0; precision = p)
    difference = BigFloat(0; precision = p)
    @inbounds for column in 1:n
        if column > 1
            row_segment = view(A, column, 1:(column - 1))
            BFLA.MA.operate!(zero, accumulator)
            for index in eachindex(row_segment)
                BFLA.MA.buffered_operate!(
                    multiplication_buffer,
                    BFLA.MA.add_mul,
                    accumulator,
                    row_segment[index],
                    row_segment[index],
                )
            end
            BFLA.MA.operate_to!(
                difference, -, A[column, column], accumulator,
            )
            diagonal = difference
        else
            diagonal = A[column, column]
        end
        diagonal <= 0 && return column
        BFLA._mpfr_sqrt!(A[column, column], diagonal)
        for row in (column + 1):n
            if column > 1
                row_segment = view(A, row, 1:(column - 1))
                diagonal_segment = view(A, column, 1:(column - 1))
                BFLA.MA.operate!(zero, accumulator)
                for index in eachindex(row_segment, diagonal_segment)
                    BFLA.MA.buffered_operate!(
                        multiplication_buffer,
                        BFLA.MA.add_mul,
                        accumulator,
                        row_segment[index],
                        diagonal_segment[index],
                    )
                end
                BFLA.MA.operate_to!(
                    difference, -, A[row, column], accumulator,
                )
                numerator = difference
            else
                numerator = A[row, column]
            end
            BFLA._mpfr_div!(
                A[row, column], numerator, A[column, column],
            )
        end
    end
    return 0
end

@testset "cholesky" begin
    for p in (128, 256, 512)
        @testset "p=$p" begin
            @testset "mutation invariant" begin
                A = make_spd(6, p)
                # allocating cholesky must preserve input bitwise
                Acopy = BFLA.owned_copy(A)
                Asnapshot = BFLA.owned_copy(Acopy)
                F = BFLA.cholesky(Native, Acopy)
                @test issuccess(F)
                @test all(Acopy[i, j] == Asnapshot[i, j] for i in 1:6, j in 1:6)
                # in-place cholesky! must modify only the lower triangle
                B = BFLA.owned_copy(A)
                Fin = BFLA.cholesky!(Native, B)
                @test issuccess(Fin)
                for j in 1:6, i in 1:(j - 1)
                    @test B[i, j] == A[i, j]
                end
            end

            @testset "1x1" begin
                A = BFLA.owned_zeros(BigFloat, 1, 1; precision_bits = p)
                A[1, 1] = BigFloat(5; precision = p)
                F = BFLA.cholesky(Native, A)
                @test issuccess(F)
                @test factor_matrix(F)[1, 1] == setprecision(BigFloat, p) do
                    sqrt(BigFloat(5; precision = p))
                end
            end

            @testset "2x2" begin
                A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
                A[1, 1] = BigFloat(4; precision = p)
                A[2, 1] = A[1, 2] = BigFloat(1; precision = p)
                A[2, 2] = BigFloat(3; precision = p)
                F = BFLA.cholesky(Native, A)
                @test issuccess(F)
                L = factor_matrix(F)
                @test L[1, 1] == BigFloat(2; precision = p)
                @test L[2, 1] == BigFloat(0.5; precision = p)
            end

            @testset "normal SPD + solve" begin
                A = make_spd(7, p)
                b = random_vector(7, p, MersenneTwister(7 + p))
                b0 = BFLA.owned_copy(b)
                Fn = BFLA.cholesky(Native, BFLA.owned_copy(A))
                Fg = BFLA.cholesky(Generic, BFLA.owned_copy(A))
                @test issuccess(Fn) && issuccess(Fg)
                xn = BFLA.owned_copy(b)
                xg = BFLA.owned_copy(b)
                BFLA.solve!(Fn, xn)
                BFLA.solve!(Fg, xg)
                assert_close(xn, xg, p; label = "cholesky solve")
                @test Float64(solve_residual(A, xn, b0)) < 1e-20
                # multiple RHS
                rhs = random_matrix(7, 3, p, MersenneTwister(70 + p))
                rhs0 = BFLA.owned_copy(rhs)
                Fn2 = BFLA.cholesky(Native, BFLA.owned_copy(A))
                BFLA.solve!(Fn2, rhs)
                @test Float64(solve_residual(A, rhs, rhs0)) < 1e-20
            end

            @testset "legacy lower trajectory" begin
                A = make_spd(6, p; seed = 8_800 + p)
                expected = BFLA.owned_copy(A)
                @test legacy_style_cholesky_lower!(expected, p) == 0
                actual = BFLA.owned_copy(A)
                factor = BFLA.cholesky!(Native, actual)
                @test issuccess(factor)
                @test actual == expected

                failed = BFLA.owned_zeros(
                    BigFloat, 2, 2; precision_bits = p,
                )
                failed[1, 1] = BigFloat(1; precision = p)
                failed[2, 1] = BigFloat(2; precision = p)
                failed[2, 2] = BigFloat(1; precision = p)
                failed_expected = BFLA.owned_copy(failed)
                @test legacy_style_cholesky_lower!(failed_expected, p) == 2
                failed_actual = BFLA.owned_copy(failed)
                failed_factor = BFLA.cholesky!(
                    Native, failed_actual; check = false,
                )
                @test factor_failure_position(failed_factor) == 2
                @test failed_actual == failed_expected
            end

            @testset "near-singular SPD" begin
                A = make_spd(5, p; delta_bits = max(p ÷ 2, 10))
                F = BFLA.cholesky(Native, BFLA.owned_copy(A))
                @test issuccess(F)
                b = random_vector(5, p, MersenneTwister(50 + p))
                BFLA.solve!(F, b)
                @test all(isfinite, b)
            end

            @testset "non-positive-definite" begin
                A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
                A[1, 1] = BigFloat(1; precision = p)
                A[2, 1] = A[1, 2] = BigFloat(2; precision = p)
                A[2, 2] = BigFloat(1; precision = p)
                @test_throws LinearAlgebra.PosDefException BFLA.cholesky!(Native, A)
                @test BFLA.try_cholesky!(Native, BFLA.owned_copy(A)) === nothing
                F = BFLA.cholesky!(Native, BFLA.owned_copy(A); check = false)
                @test !issuccess(F)
            end

            @testset "rank-deficient" begin
                A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
                A[1, 1] = BigFloat(1; precision = p)
                F = BFLA.cholesky!(Native, A; check = false)
                @test !issuccess(F)
            end

            @testset "lower finite, upper NaN" begin
                A = make_spd(5, p)
                for i in 1:5, j in (i + 1):5
                    A[i, j] = BigFloat(NaN; precision = p)
                end
                F = BFLA.cholesky(Native, BFLA.owned_copy(A))
                @test issuccess(F)
                rhs = random_vector(5, p, MersenneTwister(9000 + p))
                @test all(isfinite, BFLA.solve(F, rhs))
            end

            @testset "lower NaN fails closed" begin
                A = make_spd(5, p)
                A[3, 1] = BigFloat(NaN; precision = p)
                @test_throws DomainError BFLA.cholesky(Native, A)
                A2 = make_spd(5, p)
                A2[2, 2] = BigFloat(Inf; precision = p)
                @test_throws DomainError BFLA.cholesky(Native, A2)
            end
        end
    end


    @testset "solve non-finite and alias failures" begin
        p = 192
        F = BFLA.cholesky(Native, make_spd(3, p; seed = 9100))
        @test_throws ArgumentError BFLA.solve!(F, view(F.factors, :, 1))
        rhs = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        rhs[1] = BigFloat(Inf; precision = p)
        @test_throws DomainError BFLA.solve!(F, rhs)
        F.factors[2, 1] = BigFloat(NaN; precision = p)
        finite_rhs = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
        @test_throws DomainError BFLA.solve!(F, finite_rhs)
    end

    @testset "authoritative-triangle ownership" begin
        p = 192

        function assert_atomic_ownership_failure(backend, triangle)
            A = make_spd(4, p; seed = 9200 + Int(triangle))
            if triangle === Lower
                A[3, 1] = A[2, 1]
            else
                A[1, 3] = A[1, 2]
            end
            snapshot = BFLA.owned_copy(A)
            identities = [A[index] for index in eachindex(A)]
            @test_throws ArgumentError BFLA.cholesky!(
                backend, A; triangle = triangle,
            )
            @test all(isequal(A[index], snapshot[index]) for index in eachindex(A))
            @test all(
                A[index] === identities[position]
                for (position, index) in enumerate(eachindex(A))
            )
        end

        assert_atomic_ownership_failure(Native, Lower)
        assert_atomic_ownership_failure(Generic, Lower)
        assert_atomic_ownership_failure(Generic, Upper)

        for (backend, triangle) in (
            (Native, Lower),
            (Generic, Lower),
            (Generic, Upper),
        )
            A = make_spd(4, p; seed = 9300 + Int(triangle))
            shared = BigFloat(17; precision = p)
            poison = BigFloat(NaN; precision = p)
            if triangle === Lower
                A[1, 2] = shared
                A[1, 3] = shared
                A[1, 4] = poison
            else
                A[2, 1] = shared
                A[3, 1] = shared
                A[4, 1] = poison
            end
            @test issuccess(BFLA.cholesky!(backend, A; triangle = triangle))
        end

        # The allocating API repairs source-side sharing through owned_copy.
        diagonal = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        for i in axes(diagonal, 1)
            diagonal[i, i] = BigFloat(i + 1; precision = p)
        end
        shared_zero = BigFloat(0; precision = p)
        diagonal[2, 1] = shared_zero
        diagonal[3, 1] = shared_zero
        source_snapshot = BFLA.owned_copy(diagonal)
        F = BFLA.cholesky(Native, diagonal)
        @test issuccess(F)
        @test is_independently_owned(factor_matrix(F))
        @test all(
            isequal(diagonal[index], source_snapshot[index])
            for index in eachindex(diagonal)
        )
        @test diagonal[2, 1] === diagonal[3, 1]
    end

    @testset "diagnostics report diagonal facts" begin
        p = 192
        for (backend, triangle) in (
            (Native, Lower),
            (Generic, Lower),
            (Generic, Upper),
        )
            A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[2, 2] = BigFloat(9; precision = p)
            A[3, 3] = BigFloat(25; precision = p)
            F = BFLA.cholesky(backend, A; triangle = triangle)
            diagnostics = factor_diagnostics(F)
            @test diagnostics.factor_kind === :cholesky
            @test diagnostics.triangle === triangle
            @test diagnostics.failure_position === nothing
            @test diagnostics.min_abs_diagonal == BigFloat(2; precision = p)
            @test diagnostics.max_abs_diagonal == BigFloat(5; precision = p)
            @test diagnostics.diagonal_ratio == BigFloat(2 // 5; precision = p)

            BFLA.MA.operate!(zero, diagnostics.min_abs_diagonal)
            @test factor_diagnostics(F).min_abs_diagonal ==
                  BigFloat(2; precision = p)
        end

        failed = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        failed[1, 1] = BigFloat(1; precision = p)
        failed[2, 1] = BigFloat(2; precision = p)
        failed[2, 2] = BigFloat(1; precision = p)
        failed_factor = BFLA.cholesky!(Native, failed; check = false)
        diagnostics = factor_diagnostics(failed_factor)
        @test diagnostics.failure_position == 2
        @test diagnostics.min_abs_diagonal === nothing
        @test diagnostics.max_abs_diagonal === nothing
        @test diagnostics.diagonal_ratio === nothing

        corrupted = BFLA.cholesky(Native, begin
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[2, 2] = BigFloat(9; precision = p)
            A
        end)
        factor_matrix(corrupted)[2, 1] = BigFloat(NaN; precision = p)
        @test_throws DomainError factor_diagnostics(corrupted)

        mixed = BFLA.cholesky(Native, begin
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[2, 2] = BigFloat(9; precision = p)
            A
        end)
        factor_matrix(mixed)[2, 2] = BigFloat(3; precision = 128)
        @test_throws BFLA.PrecisionMismatch factor_diagnostics(mixed)
    end
end
