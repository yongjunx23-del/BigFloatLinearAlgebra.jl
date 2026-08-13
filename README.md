# BigFloatLinearAlgebra.jl

BigFloatLinearAlgebra (BFLA) is an independent, ownership-safe dense linear
algebra library for Julia `BigFloat` / MPFR. It provides BLAS Level 1-3 kernels,
symmetric kernels, Cholesky, symmetric-indefinite LDLT, column-pivoted QR, and
partial-pivoting LU with two auditable backends.

It is not part of SDPX and does not depend on it. SDPX is simply the first
production consumer; the public API contains no solver concepts (KKT, Schur,
cone, certificate, LP, SDP).

## Design highlights

- **Explicit backends.** Every kernel takes a `NativeBackend()` (default,
  MPFR-native) or `GenericBackend()` (reference, built on `LinearAlgebra`)
  as its first argument. `capabilities(backend)` is an audit hook.
- **Ownership-safe storage.** `BigFloat` is mutable, so `zeros(BigFloat, ...)`
  aliases one MPFR object across every slot. BFLA's `owned_zeros`,
  `owned_copy`, `convert_owned!`, `copy_owned!`, `zero_owned!`, and
  `fill_owned!` guarantee or preserve independent MPFR objects.
- **Explicit precision.** Storage and scratch are created at a traceable bit
  precision; the Native backend never inherits Julia's ambient
  `setprecision`. Mixed-precision operands fail closed.
- **No silent fallback.** A backend either supports an operation or throws
  `UnsupportedOperation`. Failures are never faked as success.
- **Stable factor protocol.** Cholesky, LDLT, RRQR, and LU expose common
  backend/precision/status/failure/diagnostic accessors, so consumers do not
  inspect concrete factor fields.
- **Explicit optional workspace.** Cholesky can reuse a caller-owned,
  precision-matched worker slot for its ownership scan. This changes neither
  backend nor numerical trajectory; other kernels do not accept an ignored
  workspace keyword.

See [docs/src](docs/src) for the frozen backend/ownership/precision contracts.

## Install

```julia
import Pkg; Pkg.add(url="https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl")
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

- Storage: `owned_zeros`, `owned_similar`, `owned_copy`, `convert_owned!`,
  `copy_owned!`, `zero_owned!`, `fill_owned!`.
- Level 1: `scal!`, `axpy!`, `axpby!`, `dot`, `norminf`.
- Level 2: `gemv!`, `trsv!`, `syr!`, `symv!`.
- Level 3: `gemm!`, `syrk!`, `gemmt!`, `syr2k!`, `trmm!`, `trsm!`.
- Cholesky: `try_cholesky!`, `cholesky!`, `cholesky`, `ldiv!`, `solve!`,
  `solve`.
- Common factor metadata: `factor_matrix`, `factor_backend`,
  `factor_precision`, `factor_status`, `factor_kind`, `factor_triangle`,
  `factor_failure_position`, `factor_diagnostics`, `issuccess`.
- LDLT: `try_ldlt!`, `ldlt!`, `ldlt`, pivot/inertia diagnostics, and solve.
- QR: `qr!`, `qr`, `applyQ!`, rank/permutation/R-diagonal diagnostics, and
  overdetermined solve.
- LU: `try_lu!`, `lu!`, `lu`, pivot/permutation diagnostics, and
  vector/multi-RHS solve.
- Numerical quality: caller-owned `residual!` and normwise backward-error
  measurement for vector and multi-RHS systems, plus explicit higher-precision
  residual diagnostics and one requested refinement correction.

Only real `BigFloat` is supported in this phase.

## Backends

| capability | Native | Generic |
| --- | --- | --- |
| GEMM/GEMV/SYRK | yes | yes |
| TRSM/TRSV/TRMM | yes | yes |
| Cholesky | lower only; explicit workspace | lower and upper; explicit workspace |
| LDLT / RRQR / LU | yes | yes |
| residual / backward error | yes | yes |
| higher-precision residual / refinement | yes | yes |
| factor solve | yes | yes |
| threading | explicit 1+ workers | single (serialized) |
| ownership safe | yes | yes |

`NativeBackend` is extracted from the SDPX legacy BigFloat dense kernels and
keeps their reduction order and MPFR ownership discipline; see
`THIRD_PARTY_NOTICES.md`. `GenericBackend` uses `LinearAlgebra` generic methods
inside a scoped, lock-guarded precision context.

## Testing and benchmarking

```julia
julia --project=. -t 1 test/runtests.jl
julia --project=. -t 4 test/runtests.jl
julia --project=. benchmark/run_production_cycles.jl
julia --project=. benchmark/run_standalone.jl
```

The unit suite covers ownership, numerical correctness (Native vs. Generic vs.
a `2p`-bit oracle), factorizations, residual/refinement, failure semantics,
precision, and concurrency. Benchmark runners report cold/warm timing,
median/IQR/min/max, allocations, effective threads, block size, and RSS only
after correctness gates. `benchmark/compare_sdpx_legacy.jl` and
`benchmark/compare_sdpx_legacy_timing.jl` are opt-in parity/A-B checks; SDPX is
never a required test dependency. A representative local report is in
[`benchmark/results/2026-08-13-round-c.md`](benchmark/results/2026-08-13-round-c.md).

## License

MIT. The Native backend derives from SDPX's MIT-licensed BigFloat kernels; see
`THIRD_PARTY_NOTICES.md`.
