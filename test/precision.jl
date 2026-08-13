using Test
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

include("test_utils.jl")

@testset "precision" begin
    @testset "Native ignores ambient global precision" begin
        p = 256
        rng = MersenneTwister(6000)
        A = random_matrix(4, 4, p, rng)
        B = random_matrix(4, 4, p, rng)
        C = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        # Force a much lower global precision; Native must not inherit it.
        setprecision(BigFloat, 64) do
            BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), C)
        end
        @test all(precision(x) == p for x in C)
        # compare against a reference computed at p bits
        Cref = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        BFLA.gemm!(Generic, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), Cref)
        assert_close(C, Cref, p; label = "precision independence")
    end

    @testset "explicit storage precision" begin
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = 512)
        @test all(precision(x) == 512 for x in A)
        B = BFLA.owned_copy(A)
        @test all(precision(x) == 512 for x in B)
    end
end
