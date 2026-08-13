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

        snapshot = BFLA.owned_copy(dst)
        identities = [dst[index] for index in eachindex(dst)]
        @test_throws BFLA.PrecisionMismatch BFLA.fill_owned!(
            dst, BigFloat(8; precision = 128),
        )
        @test all(dst[index] == snapshot[index] for index in eachindex(dst))
        @test all(
            dst[index] === identities[position]
            for (position, index) in enumerate(eachindex(dst))
        )

        mixed = BFLA.owned_copy(dst)
        mixed[3] = BigFloat(mixed[3]; precision = 128)
        mixed_snapshot = [
            BigFloat(value; precision = precision(value)) for value in mixed
        ]
        mixed_identities = [mixed[index] for index in eachindex(mixed)]
        @test_throws BFLA.PrecisionMismatch BFLA.fill_owned!(
            mixed, BigFloat(9; precision = p),
        )
        @test all(
            isequal(mixed[index], mixed_snapshot[position])
            for (position, index) in enumerate(eachindex(mixed))
        )
        @test all(
            mixed[index] === mixed_identities[position]
            for (position, index) in enumerate(eachindex(mixed))
        )
    end

    @testset "convert_owned! explicit cross-precision conversion" begin
        for q in (128, 256, 512)
            source = BFLA.owned_zeros(BigFloat, 5; precision_bits = 192)
            source[1] = BigFloat(1; precision = 64)
            source[2] = BigFloat(1 // 3; precision = 128)
            source[3] = BigFloat(-7 // 5; precision = 192)
            source[4] = BigFloat(BigInt(2)^100; precision = 256)
            source[5] = BigFloat(NaN; precision = 384)
            snapshot = [
                BigFloat(value; precision = precision(value)) for value in source
            ]

            destination = BFLA.owned_zeros(BigFloat, 5; precision_bits = q)
            identities = [destination[i] for i in eachindex(destination)]
            setprecision(BigFloat, 32) do
                @test BFLA.convert_owned!(destination, source) === destination
            end
            @test all(precision(value) == q for value in destination)
            @test all(
                destination[i] === identities[i] for i in eachindex(destination)
            )
            @test is_independently_owned(destination)
            @test all(isequal(source[i], snapshot[i]) for i in eachindex(source))
            @test all(
                isequal(destination[i], BigFloat(source[i]; precision = q))
                for i in eachindex(source)
            )
        end

        destination = BFLA.owned_zeros(BigFloat, 4; precision_bits = 256)
        source = BFLA.owned_zeros(BigFloat, 3; precision_bits = 128)
        @test_throws DimensionMismatch BFLA.convert_owned!(destination, source)
        @test_throws ArgumentError BFLA.convert_owned!(destination, destination)
        @test_throws ArgumentError BFLA.convert_owned!(
            view(destination, 1:3), view(destination, 2:4),
        )
        mixed_destination = BFLA.owned_zeros(BigFloat, 3; precision_bits = 256)
        mixed_destination[2] = BigFloat(0; precision = 128)
        @test_throws BFLA.PrecisionMismatch BFLA.convert_owned!(
            mixed_destination, source,
        )
        @test_throws ArgumentError BFLA.convert_owned!(BigFloat[], BigFloat[])

        # Separate array storage can still share mutable MPFR objects after a
        # shallow copy. In-place conversion rejects that hidden alias.
        shared_source = BFLA.owned_zeros(BigFloat, 3; precision_bits = 256)
        for i in eachindex(shared_source)
            shared_source[i] = BigFloat(i; precision = 256)
        end
        shared_destination = copy(shared_source)
        @test !Base.mightalias(shared_destination, shared_source)
        @test all(
            shared_destination[i] === shared_source[i]
            for i in eachindex(shared_source)
        )
        @test_throws ArgumentError BFLA.convert_owned!(
            shared_destination, shared_source,
        )

        # The strict copy path still rejects a cross-precision operation.
        strict_destination = BFLA.owned_zeros(BigFloat, 3; precision_bits = 256)
        @test_throws BFLA.PrecisionMismatch BFLA.copy_owned!(
            strict_destination, source,
        )

        # `copy_owned!` replaces slots, so it can safely repair object sharing
        # between otherwise distinct arrays while preserving the source.
        repaired = copy(shared_source)
        snapshot = BFLA.owned_copy(shared_source)
        @test BFLA.copy_owned!(repaired, shared_source) === repaired
        @test all(
            !(repaired[i] === shared_source[i]) for i in eachindex(repaired)
        )
        BFLA.MA.operate!(+, repaired[1], BigFloat(10; precision = 256))
        @test all(
            shared_source[i] == snapshot[i] for i in eachindex(shared_source)
        )

        overlap = BFLA.owned_zeros(BigFloat, 4; precision_bits = 256)
        for i in eachindex(overlap)
            overlap[i] = BigFloat(i; precision = 256)
        end
        overlap_snapshot = BFLA.owned_copy(overlap)
        @test_throws ArgumentError BFLA.copy_owned!(
            view(overlap, 2:4), view(overlap, 1:3),
        )
        @test all(
            overlap[i] == overlap_snapshot[i] for i in eachindex(overlap)
        )
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
