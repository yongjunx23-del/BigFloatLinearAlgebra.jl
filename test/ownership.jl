@testset "ownership" begin
    @testset "owned_zeros independence" begin
        for p in (128, 256, 512)
            A = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
            @test is_independently_owned(A)
            @test all(precision(x) == p for x in A)
        end
    end

    @testset "negative controls alias" begin
        p = 256
        Z = zeros(BigFloat, 6)
        @test !is_independently_owned(Z)
        F = fill(BigFloat(1; precision = p), 6)
        @test !is_independently_owned(F)
        A = BFLA.owned_zeros(BigFloat, 6; precision_bits = p)
        for i in 1:6
            A[i] = BigFloat(i; precision = p)
        end
        Shallow = copy(A)
        @test Shallow !== A
        @test Shallow[1] === A[1]
    end

    @testset "owned_copy is a deep numeric copy" begin
        p = 256
        A = BFLA.owned_zeros(BigFloat, 3, 3; precision_bits = p)
        for i in eachindex(A)
            A[i] = BigFloat(i; precision = p)
        end
        B = BFLA.owned_copy(A)
        @test B !== A
        @test is_independently_owned(B)
        @test all(B[i] == A[i] for i in eachindex(A))
        # mutate destination; source must not change
        B[1] = BigFloat(1000; precision = p)
        @test A[1] == BigFloat(1; precision = p)
        @test A[2] == BigFloat(2; precision = p)
        # mutate one output element; siblings must not change
        @test B[2] == BigFloat(2; precision = p)
    end

    @testset "copy_owned! / zero_owned! / fill_owned!" begin
        p = 256
        src = BFLA.owned_zeros(BigFloat, 5; precision_bits = p)
        for i in 1:5
            src[i] = BigFloat(10i; precision = p)
        end
        dst = BFLA.owned_zeros(BigFloat, 5; precision_bits = p)
        BFLA.copy_owned!(dst, src)
        @test is_independently_owned(dst)
        @test all(dst[i] == src[i] for i in 1:5)
        BFLA.zero_owned!(dst)
        @test is_independently_owned(dst)
        @test all(iszero(x) for x in dst)
        BFLA.fill_owned!(dst, BigFloat(7; precision = p))
        @test is_independently_owned(dst)
        @test all(x == BigFloat(7; precision = p) for x in dst)
    end

    @testset "factor storage semantics" begin
        p = 256
        A = make_spd(4, p)
        src = BFLA.owned_copy(A)
        # allocating cholesky must not touch source
        F = BFLA.cholesky(Native, src)
        @test issuccess(F)
        @test is_independently_owned(factor_matrix(F))
        # in-place cholesky borrows and modifies its input
        B = BFLA.owned_copy(A)
        Fin = BFLA.cholesky!(Native, B)
        @test factor_matrix(Fin) === B
        @test issuccess(Fin)
    end

    @testset "no cross-precision pollution" begin
        a128 = BFLA.owned_zeros(BigFloat, 3; precision_bits = 128)
        a256 = BFLA.owned_zeros(BigFloat, 3; precision_bits = 256)
        a512 = BFLA.owned_zeros(BigFloat, 3; precision_bits = 512)
        for (arr, p) in ((a128, 128), (a256, 256), (a512, 512))
            for i in 1:3
                arr[i] = BigFloat(i; precision = p)
            end
        end
        d128 = BFLA.dot(Native, a128, a128)
        d256 = BFLA.dot(Native, a256, a256)
        d512 = BFLA.dot(Native, a512, a512)
        @test precision(d128) == 128
        @test precision(d256) == 256
        @test precision(d512) == 512
        @test all(precision(x) == 128 for x in a128)
        @test all(precision(x) == 256 for x in a256)
        @test all(precision(x) == 512 for x in a512)
    end

    @testset "reshape alias is rejected" begin
        p = 256
        v = BFLA.owned_zeros(BigFloat, 16; precision_bits = p)
        M = reshape(v, 4, 4)
        for i in eachindex(M)
            M[i] = BigFloat(i; precision = p)
        end
        B = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        one_p = BigFloat(1; precision = p)
        zero_p = BigFloat(0; precision = p)
        # Destination M aliases source M through a reshaped view.
        @test Base.mightalias(M, @view(M[1:4, 1:4]))
        @test_throws ArgumentError BFLA.gemm!(Native, NoTrans, NoTrans, one_p, M, B, zero_p, @view(M[1:4, 1:4]))
        # A vector view aliasing the source vector must be rejected by axpy!.
        vx = BFLA.owned_zeros(BigFloat, 4; precision_bits = p)
        xv = view(vx, 1:4)
        @test_throws ArgumentError BFLA.axpy!(Native, one_p, xv, xv)

        # Positive control: independent operands must be accepted (proves the
        # alias path is what rejects the negative cases, not blanket rejection).
        A2 = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        B2 = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        C2 = BFLA.owned_zeros(BigFloat, 4, 4; precision_bits = p)
        @test !Base.mightalias(A2, C2)
        @test !Base.mightalias(B2, C2)
        @test_nowarn BFLA.gemm!(Native, NoTrans, NoTrans, one_p, A2, B2, zero_p, C2)
    end
end
