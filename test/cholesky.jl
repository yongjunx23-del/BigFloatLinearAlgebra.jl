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
end
