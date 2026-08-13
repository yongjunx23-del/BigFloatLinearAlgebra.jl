using Test
using Random
import BigFloatLinearAlgebra

const BFLA = BigFloatLinearAlgebra
const Native = BFLA.NativeBackend()
const Generic = BFLA.GenericBackend()

include("test_utils.jl")

function assert_close_scalar(x::BigFloat, ref::BigFloat, p::Int; label::AbstractString="")
    err = abs(x - ref)
    scale = abs(ref) + eps_bits(p)
    @test err <= 100 * eps_bits(p) * scale
    err <= 100 * eps_bits(p) * scale || @info "scalar mismatch" label x ref err
end

@testset "level1" begin
    for p in (128, 256, 512)
        q = 2p
        rng = MersenneTwister(1000 + p)
        n = 13
        x = random_vector(n, p, rng)
        y = random_vector(n, p, rng)
        a = random_scalar(p, rng)
        b = random_scalar(p, rng)

        @testset "dot p=$p" begin
            dn = BFLA.dot(Native, x, y)
            dg = BFLA.dot(Generic, x, y)
            @test dn == dg
            x2 = BFLA.owned_copy(x; precision_bits = q)
            y2 = BFLA.owned_copy(y; precision_bits = q)
            dref = BFLA.dot(Generic, x2, y2)
            assert_close_scalar(dn, round_precision(dref, p), p; label = "dot")
        end

        @testset "norminf p=$p" begin
            v = BFLA.owned_copy(x)
            nn = BFLA.norminf(Native, v)
            ng = BFLA.norminf(Generic, v)
            @test nn == ng
            # NaN propagation
            vnan = BFLA.owned_copy(x)
            vnan[1] = BigFloat(NaN; precision = p)
            @test isnan(BFLA.norminf(Native, vnan))
            @test isnan(BFLA.norminf(Generic, vnan))
        end

        @testset "scal! p=$p" begin
            xn = BFLA.owned_copy(x)
            BFLA.scal!(Native, a, xn)
            xg = BFLA.owned_copy(x)
            BFLA.scal!(Generic, a, xg)
            assert_close(xn, xg, p; label = "scal!")
            x2 = BFLA.owned_copy(x; precision_bits = q)
            a2 = BigFloat(a; precision = q)
            BFLA.scal!(Generic, a2, x2)
            assert_close(xn, round_precision(x2, p), p; label = "scal! ref")
        end

        @testset "axpy! p=$p" begin
            yn = BFLA.owned_copy(y)
            BFLA.axpy!(Native, a, x, yn)
            yg = BFLA.owned_copy(y)
            BFLA.axpy!(Generic, a, x, yg)
            assert_close(yn, yg, p; label = "axpy!")
            x2 = BFLA.owned_copy(x; precision_bits = q)
            y2 = BFLA.owned_copy(y; precision_bits = q)
            BFLA.axpy!(Generic, BigFloat(a; precision = q), x2, y2)
            assert_close(yn, round_precision(y2, p), p; label = "axpy! ref")
        end

        @testset "axpby! p=$p" begin
            yn = BFLA.owned_copy(y)
            BFLA.axpby!(Native, a, x, b, yn)
            yg = BFLA.owned_copy(y)
            BFLA.axpby!(Generic, a, x, b, yg)
            assert_close(yn, yg, p; label = "axpby!")
            x2 = BFLA.owned_copy(x; precision_bits = q)
            y2 = BFLA.owned_copy(y; precision_bits = q)
            BFLA.axpby!(Generic, BigFloat(a; precision = q), x2, BigFloat(b; precision = q), y2)
            assert_close(yn, round_precision(y2, p), p; label = "axpby! ref")
        end
    end
end
