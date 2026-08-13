@testset "solver-relevant symmetric kernels" begin
    operand_shape(rows::Int, columns::Int, trans::BFLA.TransposeOp) =
        trans === NoTrans ? (rows, columns) : (columns, rows)

    function assert_dimension_failure_unchanged(operation, C)
        snapshot = BFLA.owned_copy(C)
        identities = [C[index] for index in eachindex(C)]
        @test_throws DimensionMismatch operation()
        @test all(
            isequal(C[index], snapshot[index]) for index in eachindex(C)
        )
        @test all(
            C[index] === identities[position]
            for (position, index) in enumerate(eachindex(C))
        )
    end

    for p in (128, 256)
        rng = MersenneTwister(9200 + p)
        a = BigFloat(2; precision = p)
        b = BigFloat(3; precision = p)

        @testset "gemmt p=$p" begin
            for triangle in (Lower, Upper), transA in (NoTrans, Trans), transB in (NoTrans, Trans)
                n, k = 6, 4
                Am = transA === NoTrans ? (n, k) : (k, n)
                Bm = transB === NoTrans ? (k, n) : (n, k)
                A = random_matrix(Am[1], Am[2], p, rng)
                B = random_matrix(Bm[1], Bm[2], p, rng)
                C0 = random_matrix(n, n, p, rng)
                Cn = BFLA.owned_copy(C0)
                Cg = BFLA.owned_copy(C0)
                BFLA.gemmt!(Native, triangle, transA, transB, a, A, B, b, Cn)
                BFLA.gemmt!(Generic, triangle, transA, transB, a, A, B, b, Cg)
                # compare requested triangle only
                for j in 1:n
                    rngi = triangle === Lower ? (j:n) : (1:j)
                    for i in rngi
                        @test Cn[i, j] == Cg[i, j]
                    end
                end
            end
        end

        @testset "symv p=$p" begin
            n = 7
            for triangle in (Lower, Upper)
                A = random_matrix(n, n, p, rng)
                x = random_vector(n, p, rng)
                y0 = random_vector(n, p, rng)
                yn = BFLA.owned_copy(y0)
                yg = BFLA.owned_copy(y0)
                BFLA.symv!(Native, triangle, a, A, x, b, yn)
                BFLA.symv!(Generic, triangle, a, A, x, b, yg)
                @test all(yn[i] == yg[i] for i in 1:n)
                # poisoned inactive triangle must not affect the result
                A2 = BFLA.owned_copy(A)
                for j in 1:n, i in 1:n
                    if triangle === Lower && i < j
                        A2[i, j] = BigFloat(NaN; precision = p)
                    elseif triangle === Upper && i > j
                        A2[i, j] = BigFloat(NaN; precision = p)
                    end
                end
                yn2 = BFLA.owned_copy(y0)
                BFLA.symv!(Native, triangle, a, A2, x, b, yn2)
                @test all(yn2[i] == yn[i] for i in 1:n)
            end
        end

        @testset "syr2k p=$p" begin
            for triangle in (Lower, Upper), trans in (NoTrans, Trans)
                n, k = 6, 4
                Am = trans === NoTrans ? (n, k) : (k, n)
                A = random_matrix(Am[1], Am[2], p, rng)
                B = random_matrix(Am[1], Am[2], p, rng)
                C0 = random_matrix(n, n, p, rng)
                Cn = BFLA.owned_copy(C0)
                Cg = BFLA.owned_copy(C0)
                BFLA.syr2k!(Native, triangle, trans, a, A, B, b, Cn)
                BFLA.syr2k!(Generic, triangle, trans, a, A, B, b, Cg)
                for j in 1:n
                    rngi = triangle === Lower ? (j:n) : (1:j)
                    for i in rngi
                        @test Cn[i, j] == Cg[i, j]
                    end
                end
            end
        end

        @testset "capabilities include symmetric kernels" begin
            caps = BFLA.capabilities(Native)
            @test caps.gemmt && caps.symv && caps.syr2k
            @test BFLA.capabilities(Generic).gemmt && BFLA.capabilities(Generic).symv && BFLA.capabilities(Generic).syr2k
        end
    end

    @testset "complete GEMMT dimension validation" begin
        p = 128
        n, k = 5, 3
        a = BigFloat(1; precision = p)
        b = BigFloat(0; precision = p)
        rng = MersenneTwister(9901)
        for backend in (Native, Generic), triangle in (Lower, Upper),
            transA in (NoTrans, Trans), transB in (NoTrans, Trans)
            valid_A_shape = operand_shape(n, k, transA)
            valid_B_shape = operand_shape(k, n, transB)
            valid_A = random_matrix(valid_A_shape..., p, rng)
            valid_B = random_matrix(valid_B_shape..., p, rng)

            cases = (
                (
                    random_matrix(operand_shape(n - 1, k, transA)..., p, rng),
                    valid_B,
                ),
                (
                    random_matrix(operand_shape(n + 1, k, transA)..., p, rng),
                    valid_B,
                ),
                (
                    valid_A,
                    random_matrix(operand_shape(k, n - 1, transB)..., p, rng),
                ),
                (
                    valid_A,
                    random_matrix(operand_shape(k, n + 1, transB)..., p, rng),
                ),
                (
                    random_matrix(operand_shape(n, k + 1, transA)..., p, rng),
                    valid_B,
                ),
            )

            for (A, B) in cases
                C = random_matrix(n, n, p, rng)
                assert_dimension_failure_unchanged(C) do
                    BFLA.gemmt!(
                        backend, triangle, transA, transB, a, A, B, b, C,
                    )
                end
            end
        end
    end

    @testset "complete SYR2K dimension validation" begin
        p = 128
        n, k = 5, 3
        a = BigFloat(1; precision = p)
        b = BigFloat(0; precision = p)
        rng = MersenneTwister(9902)
        for backend in (Native, Generic), triangle in (Lower, Upper),
            trans in (NoTrans, Trans)
            valid_shape = operand_shape(n, k, trans)
            valid_A = random_matrix(valid_shape..., p, rng)
            valid_B = random_matrix(valid_shape..., p, rng)

            cases = (
                (
                    random_matrix(operand_shape(n - 1, k, trans)..., p, rng),
                    valid_B,
                ),
                (
                    random_matrix(operand_shape(n + 1, k, trans)..., p, rng),
                    valid_B,
                ),
                (
                    valid_A,
                    random_matrix(operand_shape(n - 1, k, trans)..., p, rng),
                ),
                (
                    valid_A,
                    random_matrix(operand_shape(n + 1, k, trans)..., p, rng),
                ),
                (
                    random_matrix(operand_shape(n, k + 1, trans)..., p, rng),
                    valid_B,
                ),
            )

            for (A, B) in cases
                C = random_matrix(n, n, p, rng)
                assert_dimension_failure_unchanged(C) do
                    BFLA.syr2k!(backend, triangle, trans, a, A, B, b, C)
                end
            end
        end
    end
end
