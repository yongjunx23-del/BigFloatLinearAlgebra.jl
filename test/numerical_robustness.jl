@testset "LDLT and RRQR numerical robustness" begin
    for p in (128, 256, 512)
        maximum_value = BFLA._with_precision(p) do
            BigFloat(floatmax(BigFloat); precision=p)
        end
        extreme = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=p)
        extreme[2, 1] = BigFloat(maximum_value; precision=p)
        extreme[1, 2] = BigFloat(maximum_value; precision=p)
        rhs = BFLA.owned_zeros(BigFloat, 2; precision_bits=p)
        rhs[1] = BigFloat(1; precision=p)
        rhs[2] = BigFloat(1; precision=p)
        for backend in (Native, Generic)
            F = BFLA.ldlt(backend, extreme)
            @test BFLA.factor_blocks(F) == [2]
            @test BFLA.factor_inertia(F) == (1, 1, 0)
            solution = BFLA.solve(F, rhs)
            expected = BigFloat(0; precision=p)
            BFLA._mpfr_div!(expected, BigFloat(1; precision=p), maximum_value)
            @test solution[1] == expected
            @test solution[2] == expected
            @test BFLA.factor_diagnostics(F).min_normalized_2x2_quality ==
                  BigFloat(1; precision=p)
        end

        delta = BigFloat(0; precision=p)
        BFLA._mpfr_set_ui_2exp!(delta, 1, -div(p, 2))
        packed = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=p)
        packed[1, 1] = BigFloat(1; precision=p)
        packed[2, 1] = BigFloat(1; precision=p)
        packed[2, 2] = BigFloat(1; precision=p)
        BFLA.MA.operate!(+, packed[2, 2], delta)
        near_factor = BFLA.BFLALDLTFactor(
            packed, Native, p, BFLA.FactorStatus(:success, nothing),
            [1, 2], [2], Bool[false, true],
        )
        xtrue = BFLA.owned_zeros(BigFloat, 2; precision_bits=p)
        xtrue[1] = BigFloat(2; precision=p)
        xtrue[2] = BigFloat(-1; precision=p)
        b = BFLA.owned_zeros(BigFloat, 2; precision_bits=p)
        BFLA.symv!(
            Native, Lower, BigFloat(1; precision=p), packed, xtrue,
            BigFloat(0; precision=p), b,
        )
        solved = BFLA.solve(near_factor, b)
        assert_close(solved, xtrue, p; label="near-singular 2x2 LDLT")

        singular = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=p)
        for row in 1:2, column in 1:2
            singular[row, column] = BigFloat(1; precision=p)
        end
        @test_throws LinearAlgebra.SingularException BFLA.ldlt(Native, singular)

        cancellation = BFLA.owned_zeros(BigFloat, 3, 2; precision_bits=p)
        cancellation[1, 1] = BigFloat(1; precision=p)
        cancellation[1, 2] = BigFloat(1; precision=p)
        epsilon = BigFloat(0; precision=p)
        BFLA._mpfr_set_ui_2exp!(epsilon, 1, -div(p, 2) - 6)
        cancellation[2, 2] = BigFloat(epsilon; precision=p)
        cancellation[3, 2] = BigFloat(epsilon; precision=p)
        Fqr = BFLA.qr(Native, cancellation)
        @test BFLA.factor_rank(Fqr) == 2
        @test BFLA.factor_jpvt(Fqr) == [1, 2]
        @test !iszero(BFLA.factor_Rdiag(Fqr)[2])
        loose = BigFloat(0; precision=p)
        BFLA._mpfr_set_ui_2exp!(loose, 1, -div(p, 2) + 2)
        @test BFLA.numerical_rank(
            Fqr; atol=loose, rtol=BigFloat(0; precision=p),
        ) == 1

        scale_probe = BFLA.owned_zeros(BigFloat, 2, 1; precision_bits=p)
        scale_probe[1, 1] = BigFloat(maximum_value; precision=p)
        @test BFLA._qr_reference_scale(scale_probe, p) == maximum_value
    end
end
