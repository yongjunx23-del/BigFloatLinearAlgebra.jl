using Test

include("test_utils.jl")

@testset "BigFloatLinearAlgebra" begin
    include("ownership.jl")
    include("level1.jl")
    include("level2.jl")
    include("level3.jl")
    include("triangular.jl")
    include("cholesky.jl")
    include("failure_semantics.jl")
    include("precision.jl")
    include("concurrency.jl")
    include("mpfr.jl")
    include("property.jl")
    include("workspace.jl")
    include("blocked.jl")
    include("threading.jl")
    include("symmetric.jl")
    include("ldlt.jl")
    include("qr.jl")
    include("quality.jl")
    include("lu.jl")
end
