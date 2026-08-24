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
            @test cache.cacheval.factor === nothing
            @test isconcretetype(typeof(cache.cacheval).parameters[1])
            sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test !cache.isfresh
            first_factor = cache.cacheval.factor
            @test first_factor isa BFLA.BFLALUFactor
            @test BFLA.norminf(Native, A * sol.u - b) <=
                BigFloat(100; precision = p) * eps(BigFloat)
            @test A == A0
            @test b == b0
            @test is_independently_owned(sol.u)
            @test all(!(sol.u[i] === cache.b[i]) for i in eachindex(sol.u))

            cache.b = BFLA.owned_copy(b)
            cache.b[1] += BigFloat(1; precision = p)
            second_sol = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(second_sol)
            @test cache.cacheval.factor === first_factor

            A2 = BFLA.owned_copy(A)
            A2[1, 1] += BigFloat(1; precision = p)
            SciMLBase.reinit!(cache; A = A2)
            @test cache.isfresh
            LinearSolve.solve!(cache)
            @test !cache.isfresh
            @test cache.cacheval.factor !== first_factor

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
            @test chol_cache.cacheval.factor isa BFLA.BFLACholeskyFactor
            chol_factor = chol_cache.cacheval.factor
            @test spd == spd0
            @test rhs == rhs0
            @test is_independently_owned(chol_sol.u)
            @test BFLA.norminf(Native, spd * chol_sol.u - rhs) <=
                BigFloat(100; precision = p) * eps(BigFloat)

            chol_cache.b = BFLA.owned_copy(rhs)
            chol_cache.b[2] += BigFloat(1; precision = p)
            second_chol_sol = LinearSolve.solve!(chol_cache)
            @test SciMLBase.successful_retcode(second_chol_sol)
            @test chol_cache.cacheval.factor === chol_factor
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
            @test cache.cacheval.factor === nothing

            good = BFLA.owned_copy(A)
            good[2, 2] = BigFloat(1; precision = p)
            SciMLBase.reinit!(cache; A = good)
            @test SciMLBase.successful_retcode(LinearSolve.solve!(cache))
            @test cache.cacheval.factor isa BFLA.BFLALUFactor

            SciMLBase.reinit!(cache; A = A)
            failed_refactor = LinearSolve.solve!(cache)
            @test failed_refactor.retcode === SciMLBase.ReturnCode.Failure
            @test cache.isfresh
            @test cache.cacheval.factor === nothing

            indefinite = BFLA.owned_copy(good)
            indefinite[2, 2] = -BigFloat(1; precision = p)
            chol_cache = LinearSolve.init(
                LinearSolve.LinearProblem(indefinite, b),
                BFLA.BigFloatCholesky(),
            )
            chol_failure = LinearSolve.solve!(chol_cache)
            @test chol_failure.retcode === SciMLBase.ReturnCode.Failure
            @test chol_cache.isfresh
            @test chol_cache.cacheval.factor === nothing
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
            @test BFLA.factor_precision(cache.cacheval.factor) == q
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

            factor = cache.cacheval.factor
            MA.operate!(+, A[1, 1], BigFloat(1; precision = p))
            @test !cache.isfresh
            LinearSolve.solve!(cache)
            @test cache.cacheval.factor === factor

            cache.A = A
            @test cache.isfresh
            refreshed = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(refreshed)
            @test cache.cacheval.factor !== factor
            @test BFLA.norminf(Native, A * refreshed.u - rhs) <=
                BigFloat(200; precision = p) * eps(BigFloat)

            refreshed_factor = cache.cacheval.factor
            MA.operate!(+, A[2, 2], BigFloat(1; precision = p))
            SciMLBase.reinit!(cache; A = A)
            @test cache.isfresh
            reinitialized = LinearSolve.solve!(cache)
            @test SciMLBase.successful_retcode(reinitialized)
            @test cache.cacheval.factor !== refreshed_factor
            @test BFLA.norminf(Native, A * reinitialized.u - rhs) <=
                BigFloat(200; precision = p) * eps(BigFloat)
        end
    end
end
