include("bench_utils.jl")

println("=== Native vs Generic backend comparison ===")
println(environment())

for p in (128, 256, 512)
    for n in (16, 64, 128)
        rng = MersenneTwister(4000 + p + n)
        A = owned_matrix(n, n, p, rng)
        B = owned_matrix(n, n, p, rng)
        C = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        one = BigFloat(1; precision = p)
        zero = BigFloat(0; precision = p)
        BFLA.gemm!(Native, NoTrans, NoTrans, one, A, B, zero, C)
        Cref = BFLA.owned_zeros(BigFloat, n, n; precision_bits = p)
        BFLA.gemm!(Generic, NoTrans, NoTrans, one, A, B, zero, Cref)
        println("p=$p n=$n max abs diff = ", maximum(abs(Float64(C[i, j] - Cref[i, j])) for i in 1:n, j in 1:n))
    end
end
