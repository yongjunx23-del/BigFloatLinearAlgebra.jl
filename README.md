# BigFloatLinearAlgebra.jl

`BigFloatLinearAlgebra.jl` (BFLA) is a dense linear-algebra library for Julia
`BigFloat` / MPFR. It provides ownership-safe storage, explicit precision,
BLAS-like kernels, and dense factorizations without depending on an
optimization solver.

The default `NativeBackend()` uses specialized MPFR-aware kernels.
`GenericBackend()` provides a reference path based on Julia's generic
`LinearAlgebra` algorithms.

## Installation

Until the package is registered in Julia General:

```julia
using Pkg
Pkg.add(url = "https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl")
```

## Quick start

```julia
using BigFloatLinearAlgebra

p = 256
backend = NativeBackend()

A = owned_zeros(BigFloat, 2, 2; precision_bits = p)
A[1, 1] = BigFloat(4; precision = p)
A[1, 2] = BigFloat(1; precision = p)
A[2, 1] = BigFloat(1; precision = p)
A[2, 2] = BigFloat(3; precision = p)

b = owned_zeros(BigFloat, 2; precision_bits = p)
b[1] = BigFloat(1; precision = p)
b[2] = BigFloat(2; precision = p)

F = cholesky(backend, A)
x = solve(F, b)
```

## Main features

- ownership-safe `BigFloat` arrays with explicit precision;
- Level 1-3 BLAS-like kernels;
- symmetric kernels and authoritative-triangle operations;
- Cholesky, symmetric-indefinite LDLT, column-pivoted QR, and LU;
- vector and multi-RHS solves;
- stable factor metadata and diagnostics;
- residual, backward-error, higher-precision residual, and refinement tools;
- caller-owned reusable solve workspace;
- explicit `NativeBackend()` and `GenericBackend()` selection;
- no silent backend fallback.

Only real `BigFloat` is supported in the current release.

## Performance

The table below compares `NativeBackend()` with `GenericBackend()`, whose
reference implementation uses Julia's generic `LinearAlgebra` path. The
measurements are correctness-gated warm medians at `n = 128` on Apple M4/macOS
with Julia 1.12.6. Values are speedups (`Generic / Native`), so larger is
better.

| Workload | 128-bit | 256-bit | 512-bit |
|---|---:|---:|---:|
| SYRK → Cholesky | 3.43x | 2.60x | 2.28x |
| Cholesky → 3 solves | 2.66x | 2.18x | 2.13x |
| TRSM → Gram → RRQR | 3.03x | 2.61x | 2.42x |
| LDLT → multi-RHS | 6.06x | 6.36x | 6.39x |

These results are machine- and workload-dependent, not portable performance
guarantees. Full timings, allocation data, correctness gates, and methodology
are in
[`benchmark/results/2026-08-13-round-c.md`](benchmark/results/2026-08-13-round-c.md).

## Precision and ownership

`BigFloat` values are mutable. BFLA therefore provides constructors and copy
operations that keep array entries independently owned and tracks the intended
bit precision explicitly.

```julia
A = owned_zeros(BigFloat, 100, 100; precision_bits = 256)
B = owned_copy(A)
```

The Native backend does not silently switch to another backend when an
operation is unsupported. Precision mismatches and unsupported operations fail
explicitly.

## Factor API

Cholesky, LDLT, QR, and LU share common public metadata such as:

```text
issuccess
factor_matrix
factor_backend
factor_precision
factor_status
factor_kind
factor_diagnostics
numerical_rank
ldiv!
solve
```

See [`docs/src`](docs/src) for the detailed ownership, precision, workspace, and
factor contracts.

## Testing

```julia
using Pkg
Pkg.test("BigFloatLinearAlgebra")
```

CI currently tests Julia 1.10, 1.11, and 1.12 with one- and four-thread runs.

## Contributors and AI disclosure

- **Yongjun Xu** — maintainer; design, implementation, numerical validation,
  benchmarking, and review.
- **OpenAI Codex** — substantial implementation, refactoring, testing, and
  documentation assistance under human review.

The Native backend also contains code extracted and generalized from the
MIT-licensed SDPX.jl BigFloat kernels. Provenance and third-party attribution
are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT. See [`LICENSE`](LICENSE).
