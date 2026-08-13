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
end
