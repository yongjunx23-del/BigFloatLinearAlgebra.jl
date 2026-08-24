using Documenter
using BigFloatLinearAlgebra

makedocs(;
    sitename = "BigFloatLinearAlgebra.jl",
    modules = [BigFloatLinearAlgebra],
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Precision" => "precision.md",
        "Ownership" => "ownership.md",
        "Backend contract" => "backend_contract.md",
        "Factor cache lifecycle" => "cache_lifecycle.md",
        "Memory accounting" => "memory_accounting.md",
        "SDPX provider contract" => "sdpx_provider_contract.md",
    ],
)
