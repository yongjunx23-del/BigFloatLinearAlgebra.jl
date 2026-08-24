using LinearSolve
import MutableArithmetics as MA
using SciMLBase

@testset "LinearSolve BigFloat extension" begin
    @test !isnothing(Base.get_extension(BFLA, :BigFloatLinearSolveExt))
    @test BFLA.BigFloatLU() isa LinearSolve.SciMLLinearSolveAlgorithm
    @test BFLA.BigFloatCholesky() isa LinearSolve.SciMLLinearSolveAlgorithm
    @test BFLA.BigFloatLU(Generic).backend === Generic
    @test BFLA.BigFloatCholesky(Generic).backend === Generic

    for p in (128, 256, 512)
        setprecision(BigFloat, p) do
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[1, 2] = BigFloat(1; precision = p)
            A[2, 1] = BigFloat(2; precision = p)
            A[2, 2] = BigFloat(3; precision = p)
            b = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
            b[1] = BigFloat(9; precision = p)
            b[2] = BigFloat(11; precision = p)
            A0 = BFLA.owned_copy(A)
            b0 = BFLA.owned_copy(b)

            prob = LinearSolve.LinearProblem(A, b)
            alg = BFLA.BigFloatLU()
            cache = LinearSolve.init(prob, alg)
            @test cache.isfresh
            @test cache.A === A
            @test cache.b === b
            @test cache.cacheval.cache === nothing
            @test isconcretetype(typeof(cache.cacheval).parameters[1])
            sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test !cache.isfresh
            first_cache = cache.cacheval.cache
            @test first_cache isa BFLA.BFLALUCache
            @test BFLA.issuccess(first_cache)
            @test BFLA.norminf(Native, A * sol.u - b) <=
                BigFloat(100; precision = p) * eps(BigFloat)
            @test A == A0
            @test b == b0
            @test is_independently_owned(sol.u)
            @test all(!(sol.u[i] === cache.b[i]) for i in eachindex(sol.u))

            # RHS-only update reuses the factor cache object.
            cache.b = BFLA.owned_copy(b)
            cache.b[1] += BigFloat(1; precision = p)
            second_sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(second_sol)
            @test cache.cacheval.cache === first_cache

            # Matrix refresh re-factorizes into the *same* cache storage; the
            # cache object identity is preserved (no factor deep-copy).
            A2 = BFLA.owned_copy(A)
            A2[1, 1] += BigFloat(1; precision = p)
            SciMLBase.reinit!(cache; A = A2)
            @test cache.isfresh
            LinearSolve.solve!(cache)
            @test !cache.isfresh
            @test cache.cacheval.cache === first_cache
            @test BFLA.issuccess(cache.cacheval.cache)

            spd = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            spd[1, 1] = BigFloat(5; precision = p)
            spd[1, 2] = spd[2, 1] = BigFloat(1; precision = p)
            spd[2, 2] = BigFloat(3; precision = p)
            rhs = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
            rhs[1] = BigFloat(7; precision = p)
            rhs[2] = BigFloat(5; precision = p)
            spd0 = BFLA.owned_copy(spd)
            rhs0 = BFLA.owned_copy(rhs)
            chol_cache = LinearSolve.init(
                LinearSolve.LinearProblem(spd, rhs), BFLA.BigFloatCholesky(),
            )
            chol_sol = LinearSolve.solve!(chol_cache)
            @test SciMLBase.successful_retcode(chol_sol)
            @test chol_cache.cacheval.cache isa BFLA.BFLACholeskyCache
            chol_cache_obj = chol_cache.cacheval.cache
            @test spd == spd0
            @test rhs == rhs0
            @test is_independently_owned(chol_sol.u)
            @test BFLA.norminf(Native, spd * chol_sol.u - rhs) <=
                BigFloat(100; precision = p) * eps(BigFloat)

            chol_cache.b = BFLA.owned_copy(rhs)
            chol_cache.b[2] += BigFloat(1; precision = p)
            second_chol_sol = LinearSolve.solve!(chol_cache)
            @test SciMLBase.successful_retcode(second_chol_sol)
            @test chol_cache.cacheval.cache === chol_cache_obj
        end
    end

    @testset "factorization failure is explicit" begin
        setprecision(BigFloat, 128) do
            p = 128
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(1; precision = p)
            A[2, 2] = BigFloat(0; precision = p)
            b = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
            cache = LinearSolve.init(
                LinearSolve.LinearProblem(A, b), BFLA.BigFloatLU(),
            )
            sol = LinearSolve.solve!(cache)
            @test sol.retcode === SciMLBase.ReturnCode.Failure
            @test cache.isfresh
            @test cache.cacheval.cache !== nothing
            @test !BFLA.issuccess(cache.cacheval.cache)

            good = BFLA.owned_copy(A)
            good[2, 2] = BigFloat(1; precision = p)
            SciMLBase.reinit!(cache; A = good)
            @test SciMLBase.successful_retcode(LinearSolve.solve!(cache))
            @test BFLA.issuccess(cache.cacheval.cache)

            SciMLBase.reinit!(cache; A = A)
            failed_refactor = LinearSolve.solve!(cache)
            @test failed_refactor.retcode === SciMLBase.ReturnCode.Failure
            @test cache.isfresh
            @test !BFLA.issuccess(cache.cacheval.cache)

            indefinite = BFLA.owned_copy(good)
            indefinite[2, 2] = -BigFloat(1; precision = p)
            chol_cache = LinearSolve.init(
                LinearSolve.LinearProblem(indefinite, b),
                BFLA.BigFloatCholesky(),
            )
            chol_failure = LinearSolve.solve!(chol_cache)
            @test chol_failure.retcode === SciMLBase.ReturnCode.Failure
            @test chol_cache.isfresh
            @test chol_cache.cacheval.cache !== nothing
            @test !BFLA.issuccess(chol_cache.cacheval.cache)
        end
    end

    @testset "explicit problem precision overrides ambient cache precision" begin
        setprecision(BigFloat, 64) do
            p = 128
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[1, 2] = BigFloat(1; precision = p)
            A[2, 1] = BigFloat(2; precision = p)
            A[2, 2] = BigFloat(3; precision = p)
            b = BFLA.owned_zeros(BigFloat, 2; precision_bits = p)
            b[1] = BigFloat(9; precision = p)
            b[2] = BigFloat(11; precision = p)

            cache = LinearSolve.init(
                LinearSolve.LinearProblem(A, b), BFLA.BigFloatLU(),
            )
            @test precision(first(cache.u)) == 64
            sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test all(precision(value) == p for value in sol.u)
            @test is_independently_owned(sol.u)

            q = 256
            A2 = BFLA.owned_copy(A; precision_bits = q)
            b2 = BFLA.owned_copy(b; precision_bits = q)
            SciMLBase.reinit!(cache; A = A2, b = b2)
            sol2 = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol2)
            @test all(precision(value) == q for value in sol2.u)
            @test BFLA.factor_precision(cache.cacheval.cache) == q
        end
    end

    @testset "matrix right-hand side and explicit cache aliasing" begin
        setprecision(BigFloat, 128) do
            p = 128
            A = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            A[1, 1] = BigFloat(4; precision = p)
            A[1, 2] = BigFloat(1; precision = p)
            A[2, 1] = BigFloat(2; precision = p)
            A[2, 2] = BigFloat(3; precision = p)
            rhs = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits = p)
            rhs[1, 1] = BigFloat(9; precision = p)
            rhs[2, 1] = BigFloat(11; precision = p)
            rhs[1, 2] = BigFloat(1; precision = p)
            rhs[2, 2] = BigFloat(5; precision = p)

            cache = LinearSolve.init(
                LinearSolve.LinearProblem(A, rhs), BFLA.BigFloatLU(),
            )
            @test size(cache.u) == size(rhs)
            sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test BFLA.norminf(Native, A * sol.u - rhs) <=
                BigFloat(200; precision = p) * eps(BigFloat)

            cache_obj = cache.cacheval.cache
            # In-place mutation of A without refresh must not refactor.
            MA.operate!(+, A[1, 1], BigFloat(1; precision = p))
            @test !cache.isfresh
            LinearSolve.solve!(cache)
            @test cache.cacheval.cache === cache_obj

            # Explicit matrix refresh re-factorizes into the same cache storage.
            cache.A = A
            @test cache.isfresh
            refreshed = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(refreshed)
            @test cache.cacheval.cache === cache_obj
            @test BFLA.issuccess(cache.cacheval.cache)
            @test BFLA.norminf(Native, A * refreshed.u - rhs) <=
                BigFloat(200; precision = p) * eps(BigFloat)

            MA.operate!(+, A[2, 2], BigFloat(1; precision = p))
            SciMLBase.reinit!(cache; A = A)
            @test cache.isfresh
            reinitialized = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(reinitialized)
            @test cache.cacheval.cache === cache_obj
            @test BFLA.norminf(Native, A * reinitialized.u - rhs) <=
                BigFloat(200; precision = p) * eps(BigFloat)
        end
    end

    @testset "RHS shape lifecycle, factor identity, and failure recovery" begin
        setprecision(BigFloat, 128) do
            p = 128
            n = 3
            A = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
            for j in 1:n, i in 1:n
                A[i, j] = BigFloat(i == j ? i + 3 : 1; precision = p)
            end
            bv = BFLA.owned_zeros(BigFloat, n; precision_bits = p)
            for i in 1:n
                bv[i] = BigFloat(i + 1; precision = p)
            end
            cache = LinearSolve.init(LinearSolve.LinearProblem(A, bv), BFLA.BigFloatLU())
            sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol)

            # Matrix refresh preserves factor-matrix and element object identity.
            cm = cache.cacheval.cache
            fm_before = BFLA.factor_matrix(cm)
            ids_before = objectid.(fm_before)
            A2 = BFLA.owned_copy(A)
            A2[1, 1] += BigFloat(1; precision = p)
            SciMLBase.reinit!(cache; A = A2)
            LinearSolve.solve!(cache)
            @test BFLA.factor_matrix(cm) === fm_before
            @test objectid.(fm_before) == ids_before

            # Failure then recovery: singular -> Failure, then good -> Success.
            As = BFLA.owned_copy(A)
            for j in 1:n
                As[2, j] = BigFloat(0; precision = p)
            end
            SciMLBase.reinit!(cache; A = As)
            fsol = LinearSolve.solve!(cache)
            @test fsol.retcode === SciMLBase.ReturnCode.Failure
            SciMLBase.reinit!(cache; A = A2)
            rsol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(rsol)
            @test BFLA.factor_matrix(cm) === fm_before
        end
    end

end
