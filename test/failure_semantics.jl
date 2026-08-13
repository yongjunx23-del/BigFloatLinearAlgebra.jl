struct FactorProbeBackend <: BFLA.AbstractBFLABackend end

@testset "failure semantics" begin
    p = 256
    rng = MersenneTwister(5000)

    @testset "dimension mismatch" begin
        A = random_matrix(3, 4, p, rng)
        B = random_matrix(4, 3, p, rng)
        C = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        @test_throws DimensionMismatch BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, A, BigFloat(0; precision = p), C)
        x = random_vector(4, p, rng)
        y = random_vector(3, p, rng)
        @test_throws DimensionMismatch BFLA.axpy!(Native, BigFloat(1; precision = p), x, y)
        @test_throws DimensionMismatch BFLA.dot(Native, x, y)
        @test_throws DimensionMismatch BFLA.cholesky!(Native, A)
    end

    @testset "destination aliases source" begin
        A = random_matrix(4, 4, p, rng)
        B = random_matrix(4, 4, p, rng)
        @test_throws ArgumentError BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), A)
        x = random_vector(4, p, rng)
        @test_throws ArgumentError BFLA.axpy!(Native, BigFloat(1; precision = p), x, x)
    end

    @testset "unsupported capability" begin
        A = make_spd(4, p)
        @test_throws BFLA.UnsupportedOperation BFLA.cholesky!(Native, BFLA.owned_copy(A); triangle = Upper)
    end

    @testset "factor operations dispatch through recorded backend" begin
        probe = FactorProbeBackend()

        spd = make_spd(3, p; seed = 5100)
        cholesky_native = BFLA.cholesky(Native, spd)
        cholesky_probe = BFLA.BFLACholeskyFactor(
            factor_matrix(cholesky_native),
            probe,
            factor_triangle(cholesky_native),
            factor_precision(cholesky_native),
            factor_status(cholesky_native),
        )
        cholesky_rhs = random_vector(3, p, MersenneTwister(5101))
        cholesky_snapshot = BFLA.owned_copy(cholesky_rhs)
        @test_throws BFLA.UnsupportedOperation BFLA.solve!(
            cholesky_probe, cholesky_rhs,
        )
        @test cholesky_rhs == cholesky_snapshot

        A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        A[1, 1] = BigFloat(2; precision = p)
        A[2, 2] = BigFloat(3; precision = p)
        ldlt_native = BFLA.ldlt(Native, A)
        ldlt_probe = BFLA.BFLALDLTFactor(
            factor_matrix(ldlt_native),
            probe,
            factor_precision(ldlt_native),
            factor_status(ldlt_native),
            copy(ldlt_native.perm),
            copy(ldlt_native.blocks),
            copy(ldlt_native.subdiag_is_d),
        )
        ldlt_rhs = random_vector(2, p, MersenneTwister(5102))
        ldlt_snapshot = BFLA.owned_copy(ldlt_rhs)
        @test_throws BFLA.UnsupportedOperation BFLA.solve!(ldlt_probe, ldlt_rhs)
        @test ldlt_rhs == ldlt_snapshot

        Qsource = random_matrix(4, 2, p, MersenneTwister(5103))
        qr_native = BFLA.qr(Native, Qsource)
        qr_probe = BFLA.BFLAQRFactor(
            factor_matrix(qr_native),
            probe,
            factor_precision(qr_native),
            factor_status(qr_native),
            qr_native.tau,
            copy(qr_native.jpvt),
            qr_native.rank,
            qr_native.tolerance,
        )
        for trans in (NoTrans, Trans)
            target = random_matrix(4, 2, p, MersenneTwister(5104 + Int(trans)))
            snapshot = BFLA.owned_copy(target)
            @test_throws BFLA.UnsupportedOperation BFLA.applyQ!(
                qr_probe, target, trans,
            )
            @test target == snapshot
        end

        qr_rhs = random_vector(4, p, MersenneTwister(5106))
        qr_snapshot = BFLA.owned_copy(qr_rhs)
        @test_throws BFLA.UnsupportedOperation BFLA.solve!(qr_probe, qr_rhs)
        @test qr_rhs == qr_snapshot

        lu_source = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        lu_source[1, 1] = BigFloat(2; precision = p)
        lu_source[2, 1] = BigFloat(1; precision = p)
        lu_source[2, 2] = BigFloat(3; precision = p)
        lu_source[3, 2] = BigFloat(1; precision = p)
        lu_source[3, 3] = BigFloat(4; precision = p)
        lu_native = BFLA.lu(Native, lu_source)
        lu_probe = BFLA.BFLALUFactor(
            factor_matrix(lu_native),
            probe,
            factor_precision(lu_native),
            factor_status(lu_native),
            copy(lu_native.pivots),
            copy(lu_native.perm),
        )
        lu_rhs = random_vector(3, p, MersenneTwister(5107))
        lu_snapshot = BFLA.owned_copy(lu_rhs)
        @test_throws BFLA.UnsupportedOperation BFLA.solve!(lu_probe, lu_rhs)
        @test lu_rhs == lu_snapshot
    end

    @testset "precision mismatch fails closed" begin
        A128 = random_matrix(3, 3, 128, MersenneTwister(1))
        B256 = random_matrix(3, 3, 256, MersenneTwister(1))
        C = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = 128), A128, B256, BigFloat(0; precision = 128), C)
    end

    @testset "factor metadata recorded" begin
        A = make_spd(4, p)
        F = BFLA.cholesky(Native, BFLA.owned_copy(A))
        @test factor_backend(F) === Native
        @test factor_precision(F) == p
        @test factor_triangle(F) === Lower
        @test issuccess(F)
        @test factor_status(F).kind === :success
        @test factor_kind(F) === :cholesky
        @test factor_failure_position(F) === nothing
    end

    @testset "common factor protocol" begin
        chol = BFLA.cholesky(Native, make_spd(3, p; seed = 5200))

        symmetric = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        symmetric[1, 1] = BigFloat(2; precision = p)
        symmetric[2, 2] = BigFloat(-3; precision = p)
        symmetric[3, 3] = BigFloat(4; precision = p)
        ldltf = BFLA.ldlt(Native, symmetric)

        qrf = BFLA.qr(
            Native, random_matrix(4, 3, p, MersenneTwister(5201)),
        )

        lusource = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        lusource[1, 1] = BigFloat(2; precision = p)
        lusource[2, 1] = BigFloat(1; precision = p)
        lusource[2, 2] = BigFloat(3; precision = p)
        lusource[3, 2] = BigFloat(1; precision = p)
        lusource[3, 3] = BigFloat(4; precision = p)
        luf = BFLA.lu(Native, lusource)

        factors = (chol, ldltf, qrf, luf)
        @test all(issuccess, factors)
        @test map(factor_backend, factors) == (Native, Native, Native, Native)
        @test map(factor_precision, factors) == (p, p, p, p)
        @test map(factor_triangle, factors) == (Lower, Lower, nothing, nothing)
        @test all(F -> factor_failure_position(F) === nothing, factors)
        @test all(F -> factor_status(F).kind === :success, factors)
        @test all(F -> factor_diagnostics(F) isa NamedTuple, factors)

        failed = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        failed[1, 1] = BigFloat(1; precision = p)
        failed[2, 1] = BigFloat(2; precision = p)
        failed[2, 2] = BigFloat(1; precision = p)
        failed_factor = BFLA.cholesky!(Native, failed; check = false)
        @test factor_failure_position(failed_factor) == 2
        @test factor_diagnostics(failed_factor).failure_position == 2
    end

    @testset "capabilities" begin
        caps = BFLA.capabilities(Native)
        @test caps.gemm && caps.gemv && caps.syrk && caps.trsm && caps.trsv
        @test caps.trmm && caps.cholesky && caps.ldlt && caps.lu
        @test caps.cholesky_workspace
        @test caps.rank_revealing_qr && caps.factor_solve
        @test !caps.unpivoted_qr
        @test caps.qr_pivoting === :column
        @test caps.least_squares_solve && caps.vector_solve && caps.multi_rhs
        @test caps.threading && caps.ownership_safe
        @test caps.precision_conversion
        # Generic supports everything Native does, plus upper-triangular Cholesky.
        gcaps = BFLA.capabilities(Generic)
        @test gcaps.gemm && gcaps.gemv && gcaps.syrk && gcaps.trsm && gcaps.trsv
        @test gcaps.trmm && gcaps.cholesky && gcaps.ldlt && gcaps.lu
        @test gcaps.cholesky_workspace
        @test gcaps.rank_revealing_qr && gcaps.factor_solve
        @test !gcaps.unpivoted_qr
        @test gcaps.qr_pivoting === :column
        @test gcaps.least_squares_solve && gcaps.vector_solve && gcaps.multi_rhs
        @test !gcaps.threading && gcaps.ownership_safe
        @test gcaps.precision_conversion
        @test !(caps == gcaps)
    end

    @testset "capabilities cholesky triangle contract" begin
        @test BFLA.capabilities(Native).cholesky_triangles == (Lower,)
        @test BFLA.capabilities(Generic).cholesky_triangles == (Lower, Upper)
        # Declaration must match behavior.
        A = make_spd(4, p)
        @test issuccess(BFLA.cholesky!(Native, BFLA.owned_copy(A); triangle = Lower, check = false))
        @test_throws BFLA.UnsupportedOperation BFLA.cholesky!(Native, BFLA.owned_copy(A); triangle = Upper, check = false)
        @test issuccess(BFLA.cholesky!(Generic, BFLA.owned_copy(A); triangle = Lower, check = false))
        @test issuccess(BFLA.cholesky!(Generic, BFLA.owned_copy(A); triangle = Upper, check = false))
        # Enum-valued capability supports direct membership checks.
        @test Lower in BFLA.capabilities(Native).cholesky_triangles
        @test !(Upper in BFLA.capabilities(Native).cholesky_triangles)
        @test Upper in BFLA.capabilities(Generic).cholesky_triangles
    end

    @testset "factor status kinds" begin
        # success
        F = BFLA.cholesky(Native, make_spd(3, p))
        @test factor_status(F).kind === :success
        @test factor_status(F).position === nothing
        # not positive definite (position = failing pivot)
        A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
        A[1, 1] = BigFloat(1; precision = p)
        A[2, 1] = A[1, 2] = BigFloat(2; precision = p)
        A[2, 2] = BigFloat(1; precision = p)
        Fbad = BFLA.cholesky!(Native, A; check = false)
        @test factor_status(Fbad).kind === :not_positive_definite
        @test factor_status(Fbad).position === 2
        # nonfinite authoritative triangle
        B = make_spd(3, p)
        B[2, 2] = BigFloat(NaN; precision = p)
        Fnan = BFLA.cholesky!(Native, B; check = false)
        @test factor_status(Fnan).kind === :nonfinite
        @test factor_status(Fnan).position === nothing
    end
end
