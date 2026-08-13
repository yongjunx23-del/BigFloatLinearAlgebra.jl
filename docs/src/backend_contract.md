# Backend contract

This document freezes the uniform semantics every BFLA operation must follow,
independent of backend. Tests in `test/` encode these rules as executable
assertions.

## Dimensions

- All dimensions are validated before a kernel runs.
- Mismatched dimensions throw `DimensionMismatch`.
- Validation happens once, outside the inner loops.
- One-based indexing is assumed for all dense kernels.

## Aliasing

- Destination operands must not alias source operands unless an operation is
  explicitly documented as in-place-only.
- `Base.mightalias` is used for the check.
- A detected unsupported alias throws `ArgumentError`.
- Within a destination array, every element must own independent MPFR storage
  (see [Ownership](ownership.md)). The library does not perform an unreliable
  runtime ownership probe on the hot path.

## Precision

- Every scratch scalar and output is created at the explicit target precision.
- `Float64` is never used as a staging type for `BigFloat` intermediates.
- Constants are built with `zero(T)`, `one(T)`, or an explicit
  `BigFloat(value; precision = p)`, never by first rounding a literal.
- Input precision is traced. A mismatch between `BigFloat` operands fails closed
  with `ArgumentError`.
- `NativeBackend` keeps every MPFR destination at the explicit target precision
  and never reads Julia's global `setprecision` context.
- `GenericBackend` computes inside a scoped, lock-guarded `setprecision` block so
  its `LinearAlgebra` generic methods also produce target-precision results.

## Triangular storage

- `syrk!` updates only the requested triangle by default; the other triangle is
  neither zeroed nor mirrored.
- `mirror_triangle!(A, triangle)` is the explicit way to complete a symmetric
  matrix.
- Cholesky and triangular solves read only the authoritative triangle. The
  non-authoritative triangle may hold stale data or `NaN` and must not affect
  the result.

## Non-finite values

- `NaN` and `Inf` inputs are never accepted into a valid factor.
- A factor that would produce `NaN` or `Inf` fails; success is never faked.
- An exception caught by a caller never silently switches the backend.

## Concurrency

- Phase 1 Native kernels are single-threaded but must be thread-safe.
- No global mutable workspace is used.
- Different precision contexts never share mutable MPFR scratch.
- The Generic backend serializes its scoped precision context through an
  internal lock; Native requires no lock.

## Failure semantics

- Unsupported operations raise `UnsupportedOperation` (or are rejected by
  `capabilities`).
- No operation-level silent fallback exists between `NativeBackend` and
  `GenericBackend`.
- `cholesky!(...; check=true)` throws `PosDefException` (or `DomainError` for
  non-finite input); `check=false` returns a factor with a nonzero `info`;
  `try_cholesky!` returns `nothing` on failure.
