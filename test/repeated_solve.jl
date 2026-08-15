@testset "trusted repeated solve and correction" begin
    function exact_solution(n, nrhs, p)
        X = BFLA.owned_zeros(BigFloat, n, nrhs; precision_bits=p)
        for column in 1:nrhs, row in 1:n
            X[row, column] = BigFloat(
                (3row - 2column) // 11; precision=p,
            )
        end
        return nrhs == 1 ? vec(X) : X
    end

    function rhs_for(A, X, p)
        one_value = BigFloat(1; precision=p)
        zero_value = BigFloat(0; precision=p)
        if X isa AbstractVector
            b = BFLA.owned_zeros(BigFloat, length(X); precision_bits=p)
            BFLA.gemv!(Native, NoTrans, one_value, A, X, zero_value, b)
            return b
        end
        B = BFLA.owned_zeros(BigFloat, size(X)...; precision_bits=p)
        BFLA.gemm!(
            Native, NoTrans, NoTrans, one_value, A, X, zero_value, B,
        )
        return B
    end

    for p in (128, 256, 512)
        n = 4
        spd = make_spd(n, p; seed=8_100 + p)
        indefinite = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for (row, column, value) in (
            (1, 2, 2), (2, 1, 2), (2, 2, 1),
            (3, 3, 3), (4, 4, -4),
        )
            indefinite[row, column] = BigFloat(value; precision=p)
        end
        square = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for column in 1:n, row in 1:n
            value = row == column ? 7 + row : row - 2column
            square[row, column] = BigFloat(value; precision=p)
        end

        for backend in (Native, Generic)
            factors = (
                BFLA.cholesky(backend, spd),
                BFLA.ldlt(backend, indefinite),
                BFLA.qr(backend, square),
                BFLA.lu(backend, square),
            )
            matrices = (spd, indefinite, square, square)
            for (F, A) in zip(factors, matrices), nrhs in (1, 3)
                Xtrue = exact_solution(n, nrhs, p)
                rhs = rhs_for(A, Xtrue, p)
                checked = BFLA.solve(F, rhs)
                trusted = BFLA.owned_copy(rhs)
                BFLA.ldiv_trusted!(F, trusted)
                @test all(isequal(checked[index], trusted[index])
                          for index in eachindex(checked, trusted))

                workspace = BFLA.BFLAWorkspace(p; workers=2)
                buffer = BFLA.workspace_buffer!(workspace, 1, 4n + 10)
                poison = BigFloat(NaN; precision=p)
                BFLA.fill_owned!(buffer, poison)
                reused = BFLA.owned_copy(rhs)
                BFLA.ldiv_trusted!(
                    F, reused; workspace=workspace, workspace_worker=1,
                )
                @test all(isequal(checked[index], reused[index])
                          for index in eachindex(checked, reused))
                @test is_independently_owned(buffer)
            end
        end
    end

    p = 256
    A = make_spd(5, p; seed=8_900)
    F = BFLA.cholesky(Native, A)
    residual = exact_solution(5, 1, p)
    expected = BFLA.solve(F, residual)
    correction = BFLA.owned_zeros(BigFloat, 5; precision_bits=p)
    workspace = BFLA.BFLAWorkspace(p; workers=1)
    BFLA.refinement_correction!(
        correction, F, residual; workspace=workspace,
    )
    @test correction == expected
    BFLA.zero_owned!(correction)
    BFLA.refinement_correction!(
        correction, F, residual; trusted=true, workspace=workspace,
    )
    @test correction == expected
    @test_throws ArgumentError BFLA.refinement_correction!(
        residual, F, residual,
    )

    rhs = exact_solution(5, 1, p)
    @test_throws BFLA.PrecisionMismatch BFLA.ldiv_trusted!(
        F, rhs; workspace=BFLA.BFLAWorkspace(128),
    )
    @test_throws ArgumentError BFLA.ldiv_trusted!(
        F, rhs; workspace=BFLA.BFLAWorkspace(p; workers=1), workspace_worker=2,
    )
    @test_throws ArgumentError BFLA.ldiv_trusted!(
        F, rhs; workspace_worker=2,
    )
    wrong_precision = BFLA.owned_zeros(BigFloat, 5; precision_bits=128)
    @test_throws BFLA.PrecisionMismatch BFLA.ldiv_trusted!(F, wrong_precision)

    setprecision(BigFloat, 64) do
        local_rhs = exact_solution(5, 1, p)
        BFLA.ldiv_trusted!(F, local_rhs; workspace=workspace)
        @test all(precision(value) == p for value in local_rhs)
    end

    concurrent_matrix = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits=p)
    for (row, column, value) in (
        (1, 2, 2), (2, 1, 2), (2, 2, 1),
        (3, 3, 3), (4, 4, -4),
    )
        concurrent_matrix[row, column] = BigFloat(value; precision=p)
    end
    concurrent_factor = BFLA.ldlt(Native, concurrent_matrix)
    concurrent_rhs = exact_solution(4, 3, p)
    concurrent_workspace = BFLA.BFLAWorkspace(p; workers=2)
    first_rhs = BFLA.owned_copy(concurrent_rhs)
    second_rhs = BFLA.owned_copy(concurrent_rhs)
    first_task = Threads.@spawn BFLA.ldiv_trusted!(
        concurrent_factor,
        first_rhs;
        workspace=concurrent_workspace,
        workspace_worker=1,
    )
    second_task = Threads.@spawn BFLA.ldiv_trusted!(
        concurrent_factor,
        second_rhs;
        workspace=concurrent_workspace,
        workspace_worker=2,
    )
    @test fetch(first_task) == fetch(second_task)
end
