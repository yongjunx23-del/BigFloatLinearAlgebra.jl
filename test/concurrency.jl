using Test
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

include("test_utils.jl")

@testset "concurrency" begin
    p = 256
    rng = MersenneTwister(7000)
    A = random_matrix(20, 20, p, rng)
    B = random_matrix(20, 20, p, rng)
    expected = BFLA.owned_zeros(BigFloat, 20, 20; precision_bits = p)
    BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), expected)

    tasks = [Threads.@spawn begin
        C = BFLA.owned_zeros(BigFloat, 20, 20; precision_bits = p)
        BFLA.gemm!(Native, NoTrans, NoTrans, BigFloat(1; precision = p), A, B, BigFloat(0; precision = p), C)
        C
    end for _ in 1:8]
    results = fetch.(tasks)

    for C in results
        @test all(precision(x) == p for x in C)
        for i in eachindex(C, expected)
            @test C[i] == expected[i]
        end
    end

    # Mix Native and Generic across tasks at a fixed precision.
    x = random_vector(20, p, rng)
    expected_dot = BFLA.dot(Native, x, x)
    mixed = [Threads.@spawn begin
        backend = isodd(i) ? Native : Generic
        BFLA.dot(backend, x, x)
    end for i in 1:8]
    for d in fetch.(mixed)
        @test d == expected_dot
    end
end
