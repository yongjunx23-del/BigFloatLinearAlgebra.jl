using Test
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

include("test_utils.jl")

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

    @testset "precision mismatch fails closed" begin
        A128 = random_matrix(3, 3, 128, MersenneTwister(1))
        B256 = random_matrix(3, 3, 256, MersenneTwister(1))
        C = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 128)
        @test_throws ArgumentError BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = 128), A128, B256, BigFloat(0; precision = 128), C)
    end

    @testset "factor metadata recorded" begin
        A = make_spd(4, p)
        F = BFLA.cholesky(Native, BFLA.owned_copy(A))
        @test factor_backend(F) === Native
        @test factor_precision(F) == p
        @test factor_triangle(F) === Lower
        @test factor_status(F) == 0
    end

    @testset "capabilities" begin
        caps = BFLA.capabilities(Native)
        @test caps.gemm && caps.gemv && caps.syrk && caps.trsm && caps.trsv
        @test caps.trmm && caps.cholesky && caps.factor_solve
        @test !caps.threading && caps.ownership_safe
        @test BFLA.capabilities(Generic) == caps
    end
end
