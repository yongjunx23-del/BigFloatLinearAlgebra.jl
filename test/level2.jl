function lower_triangular(n::Int, p::Int, rng::AbstractRNG; unit::Bool=false)
    L = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
    for j in 1:n, i in j:n
        if i == j
            L[i, j] = unit ? BigFloat(1; precision = p) : BigFloat(n + i; precision = p)
        else
            L[i, j] = BigFloat(rand(rng); precision = p)
        end
    end
    return L
end

@testset "level2" begin
    for p in (128, 256, 512)
        q = 2p
        rng = MersenneTwister(2000 + p)
        m, n = 6, 7
        A = random_matrix(m, n, p, rng)
        x = random_vector(n, p, rng)
        y0 = random_vector(m, p, rng)
        a = random_scalar(p, rng)
        b = random_scalar(p, rng)

        @testset "gemv! NoTrans p=$p" begin
            yn = BFLA.owned_copy(y0)
            BFLA.gemv!(Native, NoTrans, a, A, x, b, yn)
            yg = BFLA.owned_copy(y0)
            BFLA.gemv!(Generic, NoTrans, a, A, x, b, yg)
            assert_close(yn, yg, p; label = "gemv! N")
            A2 = BFLA.owned_copy(A; precision_bits = q)
            x2 = BFLA.owned_copy(x; precision_bits = q)
            y2 = BFLA.owned_copy(y0; precision_bits = q)
            BFLA.gemv!(Generic, NoTrans, BigFloat(a; precision = q), A2, x2, BigFloat(b; precision = q), y2)
            assert_close(yn, round_precision(y2, p), p; label = "gemv! N ref")
        end

        @testset "gemv! Trans p=$p" begin
            xt = random_vector(m, p, rng)
            yt0 = random_vector(n, p, rng)
            yn = BFLA.owned_copy(yt0)
            BFLA.gemv!(Native, Trans, a, A, xt, b, yn)
            yg = BFLA.owned_copy(yt0)
            BFLA.gemv!(Generic, Trans, a, A, xt, b, yg)
            assert_close(yn, yg, p; label = "gemv! T")
        end

        @testset "trsv! p=$p" begin
            L = lower_triangular(8, p, rng)
            rhs = random_vector(8, p, rng)
            b0 = BFLA.owned_copy(rhs)
            bn = BFLA.owned_copy(rhs)
            BFLA.trsv!(Native, Lower, NoTrans, NonUnitDiagonal, L, bn)
            bg = BFLA.owned_copy(rhs)
            BFLA.trsv!(Generic, Lower, NoTrans, NonUnitDiagonal, L, bg)
            assert_close(bn, bg, p; label = "trsv! lower")
            # backward error: L * x = b0
            r = BFLA.owned_zeros(BigFloat, 8; precision_bits = p)
            BFLA.gemv!(Native, NoTrans, BigFloat(1; precision = p), L, bn, BigFloat(0; precision = p), r)
            for i in eachindex(r)
                r[i] = r[i] - b0[i]
            end
            @test Float64(BFLA.norminf(Native, r)) < 1e-20
        end

        @testset "trsv! Trans p=$p" begin
            L = lower_triangular(8, p, rng)
            rhs = random_vector(8, p, rng)
            b0 = BFLA.owned_copy(rhs)
            bn = BFLA.owned_copy(rhs)
            BFLA.trsv!(Native, Lower, Trans, NonUnitDiagonal, L, bn)
            bg = BFLA.owned_copy(rhs)
            BFLA.trsv!(Generic, Lower, Trans, NonUnitDiagonal, L, bg)
            assert_close(bn, bg, p; label = "trsv! trans")
        end

        @testset "syr! p=$p" begin
            v = random_vector(6, p, rng)
            A0 = random_matrix(6, 6, p, rng)
            An = BFLA.owned_copy(A0)
            BFLA.syr!(Native, Lower, a, v, An)
            Ag = BFLA.owned_copy(A0)
            BFLA.syr!(Generic, Lower, a, v, Ag)
            # compare lower triangles
            for j in 1:6, i in j:6
                @test An[i, j] == Ag[i, j]
            end
        end
    end
end
