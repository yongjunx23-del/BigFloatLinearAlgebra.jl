@testset "workspace and config" begin
    @testset "KernelConfig" begin
        cfg = BFLA.KernelConfig()
        @test cfg.thread_count == 1
        @test cfg.gemm_block == 0 && cfg.syrk_block == 0
        @test cfg.cholesky_block == 0 && cfg.trsm_block == 0
        cfg2 = BFLA.KernelConfig(thread_count = 4, gemm_block = 32)
        @test cfg2.thread_count == 4 && cfg2.gemm_block == 32
        @test_throws ArgumentError BFLA.KernelConfig(bogus = 1)
        @test_throws ArgumentError BFLA.KernelConfig(thread_count = 0)
        @test_throws ArgumentError BFLA.KernelConfig(thread_count = -1)
        for field in (
            :gemm_block,
            :syrk_block,
            :cholesky_block,
            :trsm_block,
        )
            @test_throws ArgumentError BFLA.KernelConfig(; field => -1)
        end
        @test_throws ArgumentError BFLA.KernelConfig(ldlt_block = 1)
        @test_throws ArgumentError BFLA.KernelConfig(0, 0, 0, 0, 0)
    end

    @testset "BFLAWorkspace ownership and precision" begin
        ws = BFLA.BFLAWorkspace(256; workers = 3, scalar_slots = 8)
        @test BFLA.workspace_precision(ws) == 256
        @test BFLA.workspace_workers(ws) == 3
        # scratch is worker-local and owned
        s1 = BFLA.workspace_scratch!(ws, 1, 2)
        s2 = BFLA.workspace_scratch!(ws, 2, 2)
        @test precision(s1) == 256 && precision(s2) == 256
        @test s1 !== s2
        @test_throws ArgumentError BFLA.workspace_scratch!(ws, 4, 1)
        @test_throws ArgumentError BFLA.workspace_scratch!(ws, 1, 99)
        # buffer is worker-local, independently owned, explicit precision
        b = BFLA.workspace_buffer!(ws, 2, 64)
        @test length(b) == 64
        @test all(precision(x) == 256 for x in b)
        @test is_independently_owned(b)
        # growing preserves precision and ownership
        b2 = BFLA.workspace_buffer!(ws, 2, 128)
        @test length(b2) == 128
        @test all(precision(x) == 256 for x in b2)
        @test is_independently_owned(b2)

        A = random_matrix(2, 2, 256, MersenneTwister(5200))
        B = random_matrix(2, 2, 256, MersenneTwister(5201))
        C = random_matrix(2, 2, 256, MersenneTwister(5202))
        one_value = BigFloat(1; precision = 256)
        zero_value = BigFloat(0; precision = 256)
        @test_throws MethodError BFLA.gemm!(
            Native,
            NoTrans,
            NoTrans,
            one_value,
            A,
            B,
            zero_value,
            C;
            workspace = ws,
        )
    end

    @testset "Cholesky ownership workspace" begin
        p = 256
        n = 12
        ws = BFLA.BFLAWorkspace(p; workers = 2)

        for precision_bits in (128, 256, 512)
            precision_source = make_spd(
                5, precision_bits; seed = 5_250 + precision_bits,
            )
            precision_reference = BFLA.cholesky(
                Native, precision_source,
            )
            precision_factor = BFLA.cholesky(
                Native,
                precision_source;
                workspace = BFLA.BFLAWorkspace(precision_bits),
            )
            @test factor_precision(precision_factor) == precision_bits
            @test all(
                isequal(
                    factor_matrix(precision_factor)[index],
                    factor_matrix(precision_reference)[index],
                )
                for index in eachindex(factor_matrix(precision_factor))
            )
        end

        reference = BFLA.cholesky(
            Native, make_spd(n, p; seed = 5_300),
        )

        buffer = BFLA._workspace_identity_buffer(ws, 1, p, "test")
        @test isempty(buffer)
        A = make_spd(n, p; seed = 5_300)
        factor = BFLA.cholesky!(Native, A; workspace = ws)
        @test issuccess(factor)
        @test buffer === BFLA._workspace_identity_buffer(ws, 1, p, "test")
        @test length(buffer) == n * (n + 1) ÷ 2
        @test all(
            isequal(factor_matrix(factor)[index], factor_matrix(reference)[index])
            for index in eachindex(factor_matrix(factor))
        )

        # A second call reuses the same vector without changing the numerical
        # trajectory. The allocating API also forwards the explicit workspace.
        source = make_spd(n, p; seed = 5_301)
        source_snapshot = BFLA.owned_copy(source)
        allocated_factor = BFLA.cholesky(
            Native, source; workspace = ws, workspace_worker = 1,
        )
        @test issuccess(allocated_factor)
        @test buffer === BFLA._workspace_identity_buffer(ws, 1, p, "test")
        @test all(
            isequal(source[index], source_snapshot[index])
            for index in eachindex(source)
        )
        @test BFLA.try_cholesky!(
            Native,
            BFLA.owned_copy(source);
            workspace = ws,
            workspace_worker = 1,
        ) !== nothing

        # Generic upper Cholesky uses the same backend-independent ownership
        # precheck while retaining its own factorization path.
        upper_source = make_spd(7, p; seed = 5_302)
        upper_reference = BFLA.cholesky(
            Generic, upper_source; triangle = Upper,
        )
        upper_factor = BFLA.cholesky(
            Generic,
            upper_source;
            triangle = Upper,
            workspace = ws,
            workspace_worker = 2,
        )
        @test all(
            isequal(
                factor_matrix(upper_factor)[index],
                factor_matrix(upper_reference)[index],
            )
            for index in eachindex(factor_matrix(upper_factor))
        )

        function assert_unchanged_after_failure(
            f, exception_type::Type, matrix,
        )
            snapshot = BFLA.owned_copy(matrix)
            identities = [matrix[index] for index in eachindex(matrix)]
            @test_throws exception_type f()
            @test all(
                isequal(matrix[index], snapshot[index])
                for index in eachindex(matrix)
            )
            @test all(
                matrix[index] === identities[position]
                for (position, index) in enumerate(eachindex(matrix))
            )
        end

        wrong_precision = make_spd(5, p; seed = 5_303)
        assert_unchanged_after_failure(
            BFLA.PrecisionMismatch, wrong_precision,
        ) do
            BFLA.cholesky!(
                Native,
                wrong_precision;
                workspace = BFLA.BFLAWorkspace(128),
            )
        end
        @test_throws BFLA.PrecisionMismatch BFLA.cholesky!(
            Native,
            BFLA.owned_copy(wrong_precision);
            workspace = BFLA.BFLAWorkspace(128),
        )

        invalid_worker = make_spd(5, p; seed = 5_304)
        assert_unchanged_after_failure(ArgumentError, invalid_worker) do
            BFLA.cholesky!(
                Native, invalid_worker; workspace = ws, workspace_worker = 3,
            )
        end
        no_workspace = make_spd(5, p; seed = 5_305)
        assert_unchanged_after_failure(ArgumentError, no_workspace) do
            BFLA.cholesky!(Native, no_workspace; workspace_worker = 2)
        end

        aliased = make_spd(5, p; seed = 5_306)
        aliased[4, 1] = aliased[3, 1]
        assert_unchanged_after_failure(ArgumentError, aliased) do
            BFLA.cholesky!(Native, aliased; workspace = ws)
        end

        # Distinct worker slots can be used by concurrent tasks. The one-thread
        # test run checks scheduling; the four-thread run exercises overlap.
        concurrent_source = make_spd(10, p; seed = 5_307)
        first_matrix = BFLA.owned_copy(concurrent_source)
        second_matrix = BFLA.owned_copy(concurrent_source)
        first_task = Threads.@spawn BFLA.cholesky!(
            Native, first_matrix; workspace = ws, workspace_worker = 1,
        )
        second_task = Threads.@spawn BFLA.cholesky!(
            Native, second_matrix; workspace = ws, workspace_worker = 2,
        )
        first_factor = fetch(first_task)
        second_factor = fetch(second_task)
        @test all(
            isequal(
                factor_matrix(first_factor)[index],
                factor_matrix(second_factor)[index],
            )
            for index in eachindex(factor_matrix(first_factor))
        )
    end
end
