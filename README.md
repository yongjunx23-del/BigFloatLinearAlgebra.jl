# BigFloatLinearAlgebra.jl

BigFloatLinearAlgebra (BFLA) is an independent, ownership-safe dense linear
algebra library for Julia `BigFloat` / MPFR. It provides BLAS Level 1-3 kernels
and a Cholesky factorization/solve with two auditable backends.

It is not part of SDPX and does not depend on it. SDPX is simply the first
production consumer; the public API contains no solver concepts (KKT, Schur,
cone, certificate, LP, SDP).

## Design highlights

- **Explicit backends.** Every kernel takes a `NativeBackend()` (default,
  MPFR-native) or `GenericBackend()` (reference, built on `LinearAlgebra`)
  as its first argument. `capabilities(backend)` is an audit hook.
- **Ownership-safe storage.** `BigFloat` is mutable, so `zeros(BigFloat, ...)`
  aliases one MPFR object across every slot. BFLA's `owned_zeros`,
  `owned_copy`, `copy_owned!`, `zero_owned!`, and `fill_owned!` guarantee
  independent MPFR objects.
- **Explicit precision.** Storage and scratch are created at a traceable bit
  precision; the Native backend never inherits Julia's ambient
  `setprecision`. Mixed-precision operands fail closed.
- **No silent fallback.** A backend either supports an operation or throws
  `UnsupportedOperation`. Failures are never faked as success.

See [docs/src](docs/src) for the frozen backend/ownership/precision contracts.

## Install

```julia
import Pkg; Pkg.add(url="<path-to-this-repository>")
```

## Quick start

```julia
using BigFloatLinearAlgebra

p = 256
A = owned_zeros(BigFloat, 4, 4; precision_bits = p)
fill_owned!(A, BigFloat(1; precision = p))
for i in 1:4
    A[i, i] = BigFloat(10; precision = p)
end

backend = NativeBackend()
C = owned_zeros(BigFloat, 4, 4; precision_bits = p)
gemm!(backend, Trans, NoTrans, BigFloat(1; precision = p), A, A,
      BigFloat(0; precision = p), C)

factor = cholesky(backend, C)
b = owned_zeros(BigFloat, 4; precision_bits = p)
b[1] = BigFloat(1; precision = p)
solve!(factor, b)
```

## Public API

- Storage: `owned_zeros`, `owned_similar`, `owned_copy`, `copy_owned!`,
  `zero_owned!`, `fill_owned!`.
- Level 1: `scal!`, `axpy!`, `axpby!`, `dot`, `norminf`.
- Level 2: `gemv!`, `trsv!`, `syr!`.
- Level 3: `gemm!`, `syrk!`, `trmm!`, `trsm!`.
- Cholesky: `try_cholesky!`, `cholesky!`, `cholesky`, `ldiv!`, `solve!`,
  `solve`, `factor_matrix`, `factor_backend`, `factor_triangle`,
  `factor_precision`, `factor_status`, `issuccess`.

Only real `BigFloat` is supported in this phase.

## Backends

| capability | Native | Generic |
| --- | --- | --- |
| GEMM/GEMV/SYRK | yes | yes |
| TRSM/TRSV/TRMM | yes | yes |
| Cholesky | lower only | lower and upper |
| factor solve | yes | yes |
| threading | single (safe) | single (serialized) |
| ownership safe | yes | yes |

`NativeBackend` is extracted from the SDPX legacy BigFloat dense kernels and
keeps their reduction order and MPFR ownership discipline; see
`THIRD_PARTY_NOTICES.md`. `GenericBackend` uses `LinearAlgebra` generic methods
inside a scoped, lock-guarded precision context.

## Testing and benchmarking

```julia
julia --project=. test/runtests.jl
julia --project=. benchmark/run_kernels.jl
```

The unit suite covers ownership, numerical correctness (Native vs. Generic vs.
a `2p`-bit oracle), Cholesky fixtures, failure semantics, precision, and
concurrency. The benchmark runner reports median/IQR/min/max and allocations
after a correctness gate, and `benchmark/compare_sdpx_legacy.jl` is an opt-in
SDPX parity check.

## License

MIT. The Native backend derives from SDPX's MIT-licensed BigFloat kernels; see
`THIRD_PARTY_NOTICES.md`.
