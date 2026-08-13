@testset "final-review regressions" begin
    @testset "Bunch-Kaufman intermediate keep-k criterion" begin
        for p in (128, 256, 512), backend in (Native, Generic)
            A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
            function set_sym!(i, j, value)
                BFLA.MA.operate_to!(A[i, j], copy, value)
                BFLA.MA.operate_to!(A[j, i], copy, value)
            end
            set_sym!(1, 1, BigFloat(1//2; precision = p))
            set_sym!(2, 1, BigFloat(1; precision = p))
            set_sym!(2, 2, BigFloat(2; precision = p))
            set_sym!(3, 1, BigFloat(0; precision = p))
            set_sym!(3, 2, BigFloat(4; precision = p))
            set_sym!(3, 3, BigFloat(1; precision = p))

            F = BFLA.ldlt(backend, A)
            @test issuccess(F)
            @test first(factor_blocks(F)) == 1

            rhs = random_vector(3, p, MersenneTwister(1750 + p))
            rhs0 = BFLA.owned_copy(rhs)
            BFLA.solve!(F, rhs)
            r = BFLA.owned_zeros(BigFloat, 3; precision_bits = p)
            BFLA.residual!(backend, A, rhs, rhs0, r)
            @test BFLA.norminf(backend, r) <=
                  BigFloat(100; precision = p) * eps_bits(p)
        end
    end

    @testset "unsupported configurable dispatch is identifiable and atomic" begin
        probe = FactorProbeBackend()
        p = 192
        a = BigFloat(1; precision = p)
        z = BigFloat(0; precision = p)
        A = make_spd(3, p)
        B = random_matrix(3, 3, p, MersenneTwister(5151))

        function assert_unchanged_after_unsupported(call, destination)
            snapshot = BFLA.owned_copy(destination)
            identities = [destination[i] for i in eachindex(destination)]
            @test_throws BFLA.UnsupportedOperation call()
            @test destination == snapshot
            @test all(
                destination[i] === identities[j]
                for (j, i) in enumerate(eachindex(destination))
            )
        end

        C = random_matrix(3, 3, p, MersenneTwister(5152))
        assert_unchanged_after_unsupported(C) do
            BFLA.gemm!(probe, NoTrans, NoTrans, a, A, B, z, C)
        end

        S = random_matrix(3, 3, p, MersenneTwister(5153))
        assert_unchanged_after_unsupported(S) do
            BFLA.syrk!(probe, Lower, NoTrans, a, A, z, S)
        end

        T = random_matrix(3, 2, p, MersenneTwister(5154))
        assert_unchanged_after_unsupported(T) do
            BFLA.trsm!(
                probe, LeftSide, Lower, NoTrans, NonUnitDiagonal, a, A, T,
            )
        end

        Cchol = BFLA.owned_copy(A)
        assert_unchanged_after_unsupported(Cchol) do
            BFLA.cholesky!(probe, Cchol)
        end

        L = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        L[1, 1] = BigFloat(2; precision = p)
        L[2, 1] = BigFloat(1; precision = p)
        L[2, 2] = BigFloat(3; precision = p)
        L[3, 2] = BigFloat(1; precision = p)
        L[3, 3] = BigFloat(4; precision = p)
        L[1, 2] = BigFloat(999; precision = p)
        L[1, 3] = BigFloat(777; precision = p)
        L[2, 3] = BigFloat(555; precision = p)
        assert_unchanged_after_unsupported(L) do
            BFLA.ldlt!(probe, L)
        end
    end
end
