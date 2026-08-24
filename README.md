# BigFloatLinearAlgebra.jl

`BigFloatLinearAlgebra.jl` (BFLA) provides dense linear algebra for Julia
`BigFloat` / MPFR. It includes ownership-safe storage, explicit precision,
BLAS-like kernels, and dense factorizations without depending on an
optimization solver.

`NativeBackend()` uses MPFR-aware kernels. `GenericBackend()` is the reference
implementation based on Julia's generic `LinearAlgebra` algorithms. Backend
selection is explicit and operations never silently fall back.

## Installation

```julia
using Pkg
Pkg.add("BigFloatLinearAlgebra")
```

## Example

```julia
using BigFloatLinearAlgebra

p = 256
A = owned_zeros(BigFloat, 2, 2; precision_bits = p)
A[1, 1] = BigFloat(4; precision = p)
A[1, 2] = BigFloat(1; precision = p)
A[2, 1] = BigFloat(1; precision = p)
A[2, 2] = BigFloat(3; precision = p)

b = owned_zeros(BigFloat, 2; precision_bits = p)
b[1] = BigFloat(1; precision = p)
b[2] = BigFloat(2; precision = p)

F = cholesky(NativeBackend(), A)
x = solve(F, b)
```

## Features

- ownership-safe `BigFloat` arrays at explicit precision;
- Level 1-3 and symmetric BLAS-like kernels;
- Cholesky, symmetric-indefinite LDLT, column-pivoted QR, and LU;
- vector and multi-RHS factor solves;
- optional LinearSolve.jl LU and Cholesky algorithms with factor reuse;
- factor metadata, residuals, backward error, and refinement helpers;
- caller-owned, worker-local workspace for repeated solves;
- Native and Generic backends with no hidden fallback.

Only real `BigFloat` is currently supported. Precision mismatches,
unsupported aliasing, non-finite factor inputs, and unsupported backend
operations fail explicitly. See [`docs/src`](docs/src) for the complete API,
ownership, precision, workspace, and backend contracts.

## Reusable factor cache

For repeated factor/solve cycles, use a precision-specific factor cache. The
owned factor storage and scratch are reserved once by `prepare!` and reused, so
the hot path allocates no new Julia `BigFloat` objects.

```julia
using BigFloatLinearAlgebra
import MutableArithmetics as MA

n, p = 32, 256
# Build an SPD n×n matrix A and an n-vector RHS b at precision p.
A = owned_zeros(BigFloat, n, n; precision_bits = p)
for j in 1:n, i in 1:n
    A[i, j] = BigFloat(i == j ? n - i + 8 : (i + j) % 5 + 1; precision = p)
end
for i in 1:n
    MA.operate!(+, A[i, i], BigFloat(n; precision = p))  # ensure SPD diagonal
end
b = owned_zeros(BigFloat, n; precision_bits = p)
for i in 1:n
    b[i] = BigFloat(i + 1; precision = p)
end

cache = BFLACholeskyCache(NativeBackend())
prepare!(cache, n, p; nrhs = 1)          # reserve owned storage (allocating)
factorize!(cache, A)                     # factor A into the cache's owned matrix
x = owned_zeros(BigFloat, n; precision_bits = p)
solve_trusted!(x, cache, b)              # trusted: x must be owned at precision p
# checked alternative for an arbitrary destination (re-owns safely):
x2 = owned_zeros(BigFloat, n; precision_bits = p)
fill!(x2, BigFloat(0; precision = p))
solve!(x2, cache, b)                     # re-owns x2 safely (allocates by design)
invalidate!(cache)                       # drop the factorization, keep storage
```

`solve!(x, cache, b)` is the **checked** API: it validates and re-owns the
destination, so it is safe for any array but allocates. `solve_trusted!(x,
cache, b)` is the **solver-facing hot path**: it requires `x` to be
independently owned at the factor precision (e.g. from `owned_zeros`) and writes
into the existing elements, allocating nothing. See
[`docs/src/cache_lifecycle.md`](docs/src/cache_lifecycle.md).

## Testing

```julia
using Pkg
Pkg.test("BigFloatLinearAlgebra")
```

CI tests Julia 1.10, 1.11, and 1.12 with one and four Julia threads.
Correctness-gated benchmark programs and reproducibility notes are in
[`benchmark`](benchmark).

## Contributors and AI disclosure

- **Yongjun Xu** is the maintainer and owns the design, numerical validation,
  review, and release decisions.
- **OpenAI Codex** made substantial implementation, refactoring, test, and
  documentation contributions under human review.

The Native backend includes code extracted and generalized from the
MIT-licensed SDPX.jl BigFloat kernels. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance.

## License

MIT. See [`LICENSE`](LICENSE).
