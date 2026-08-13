@testset "mpfr wrappers" begin
    for p in (128, 256, 512)
        dst = BigFloat(0; precision = p)
        num = BigFloat("1.234567890123456789012345678901234567890123456789"; precision = p)
        den = BigFloat("3.141592653589793238462643383279502884197169399375"; precision = p)
        BFLA._mpfr_div!(dst, num, den)
        @test dst == setprecision(BigFloat, p) do
            num / den
        end
        @test precision(dst) == p

        sqrt_dst = BigFloat(0; precision = p)
        BFLA._mpfr_sqrt!(sqrt_dst, num)
        @test sqrt_dst == setprecision(BigFloat, p) do
            sqrt(num)
        end
        @test precision(sqrt_dst) == p

        half = BigFloat(0; precision = p)
        BFLA._mpfr_div_2!(half, num)
        @test half == setprecision(BigFloat, p) do
            num / 2
        end
    end
end
