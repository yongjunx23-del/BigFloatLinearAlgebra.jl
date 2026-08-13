@testset "triangular" begin
    for p in (128, 256)
        rng = MersenneTwister(4000 + p)

        @testset "mirror_triangle! Lower p=$p" begin
            A = random_matrix(5, 5, p, rng)
            BFLA.mirror_triangle!(A, Lower)
            @test all(A[i, j] == A[j, i] for i in 1:5, j in 1:5)
            @test is_independently_owned(A)
        end

        @testset "mirror_triangle! Upper p=$p" begin
            A = random_matrix(5, 5, p, rng)
            BFLA.mirror_triangle!(A, Upper)
            @test all(A[i, j] == A[j, i] for i in 1:5, j in 1:5)
        end

        @testset "syrk! leaves other triangle untouched p=$p" begin
            A = random_matrix(5, 6, p, rng)
            C0 = random_matrix(5, 5, p, rng)
            marker = BigFloat(12345; precision = p)
            C = BFLA.owned_copy(C0)
            # fill the upper triangle with a recognizable marker
            for j in 1:5, i in 1:(j - 1)
                C[i, j] = marker
            end
            BFLA.syrk!(Native, Lower, NoTrans, BigFloat(1; precision = p), A, BigFloat(0; precision = p), C)
            for j in 1:5, i in 1:(j - 1)
                @test C[i, j] == marker
            end
        end
    end
end
