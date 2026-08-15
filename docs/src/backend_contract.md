# Backend contract

This document freezes the uniform semantics every BFLA operation must follow,
independent of backend. Tests in `test/` encode these rules as executable
assertions.

## Dimensions

- All dimensions are validated before a kernel runs.
- Mismatched dimensions throw `DimensionMismatch`.
- Validation happens once, outside the inner loops.
- `gemmt!` and `syr2k!` validate both outer dimensions and the complete
  contraction dimensions of their transformed operands before touching the
  destination. A dimension failure leaves the destination unchanged.
- One-based indexing is assumed for all dense kernels.
- `residual!` accepts vector or matrix (multi-RHS) operands with matching
  dimensionality; its caller-provided destination has the same shape as `b`.

## Aliasing

- Destination operands must not alias source operands unless an operation is
  explicitly documented as in-place-only.
- `Base.mightalias` is used for the check.
- A detected unsupported alias throws `ArgumentError`.
- Ownership conversion also accounts for cross-array MPFR object sharing that
  `Base.mightalias` cannot see: `convert_owned!` rejects it, while
  `copy_owned!` safely breaks it by installing fresh destination objects.
- Within a destination array, every element must own independent MPFR storage
  (see [Ownership](ownership.md)). The library does not perform an unreliable
  runtime ownership probe on the hot path.

## Precision

- Every scratch scalar and output is created at the explicit target precision.
- `Float64` is never used as a staging type for `BigFloat` intermediates.
- Constants are built with `zero(T)`, `one(T)`, or an explicit
  `BigFloat(value; precision = p)`, never by first rounding a literal.
- Input precision is traced across every element, not just the first. An
  intra-array or cross-operand mismatch fails closed with `PrecisionMismatch`.
- `fill_owned!` requires the fill value and every destination element to have
  one precision. It completes all precision validation before replacing any
  destination element, so a mismatch leaves the destination unchanged.
- `NativeBackend` keeps every MPFR destination at the explicit target precision
  and never reads Julia's global `setprecision` context.
- `GenericBackend` computes inside a scoped, lock-guarded `setprecision` block so
  its `LinearAlgebra` generic methods also produce target-precision results.

## Triangular storage

- `syrk!` updates only the requested triangle by default; the other triangle is
  neither zeroed nor mirrored.
- `mirror_triangle!(A, triangle)` is the explicit way to complete a symmetric
  matrix. It validates the precision of every matrix element before copying;
  a precision mismatch leaves the matrix unchanged.
- Cholesky and triangular solves read only the authoritative triangle. The
  non-authoritative triangle may hold stale data or `NaN` and must not affect
  the result.

## Non-finite values

- `NaN` and `Inf` inputs are never accepted into a valid factor.
- A factor that would produce `NaN` or `Inf` fails; success is never faked.
- An exception caught by a caller never silently switches the backend.

## Concurrency

- Native kernels are thread-safe. Supported Level-3 paths use the explicit
  `KernelConfig.thread_count`; the conservative default remains one worker.
- No global mutable workspace is used.
- Worker-local scratch is independently owned, and different precision
  contexts never share mutable MPFR scratch.
- `BFLAWorkspace` is caller-managed storage. Cholesky and LDLT factorization
  can reuse a `UInt` identity buffer for ownership prechecks. Cholesky, LDLT,
  RRQR, and LU solves accept its validated worker contract. Native solve
  scratch and the common LDLT/RRQR buffers are reused; Generic Cholesky/LU
  retain `LinearAlgebra` internal allocation. Workspace precision must match
  factor precision, scratch may be overwritten, and concurrent calls sharing
  a workspace use distinct worker slots.
- The Generic backend serializes its scoped precision context through an
  internal lock; Native requires no lock.

## Failure semantics

- Unsupported operations raise `UnsupportedOperation` (or are rejected by
  `capabilities`). `capabilities(backend).cholesky_triangles` reports exactly
  which Cholesky triangles the backend supports using the public `Triangle`
  enum: `(Lower,)` for `NativeBackend` and `(Lower, Upper)` for
  `GenericBackend`, so callers can test membership directly without symbol
  translation.
- No operation-level silent fallback exists between `NativeBackend` and
  `GenericBackend`.
- QR means column-pivoted rank-revealing Householder QR (`A*P = Q*R`). Backend
  capabilities explicitly report `unpivoted_qr = false`,
  `rank_revealing_qr = true`, and `qr_pivoting = :column`.
- RRQR rank uses `max(atol, rtol*reference_scale)` with the largest original
  input-column 2-norm as `reference_scale`. The reference scale and effective
  threshold are factor metadata, so callers can re-evaluate rank without
  reading packed `R` fields or relying on ambient precision.
- Every RRQR use boundary validates the packed matrix, Householder coefficients,
  and rank-policy metadata at the recorded factor precision. Non-finite or
  mixed-precision auxiliary metadata is rejected before caller-owned output is
  modified, including before `refine_once!` writes residual or correction
  storage.
- LDLT solve, QR Q application, and QR solve dispatch through the backend
  recorded in the factor. An unsupported recorded backend raises
  `UnsupportedOperation` before numerical storage is modified.
- The checked `ldiv!` remains the default factor-use boundary and validates
  live factor storage and metadata. `ldiv_trusted!` is a separate explicit API
  for callers that guarantee those objects have not changed since successful
  factorization. Trusted solve still validates status, dimensions, RHS alias,
  RHS precision and finiteness, workspace, and recorded backend dispatch. It
  never retries or falls back.
- Residual and backward-error primitives report numerical facts only. They do
  not choose a tolerance, refinement count, precision escalation, factorization,
  or backend fallback.
- Higher-precision residual evaluation is only available through the explicit
  `higher_precision_residual!` API. It requires an explicit
  `residual_precision=q`, matching q-bit caller storage, with `q > p`, and
  returns both factor and residual precision in its diagnostics.
- `refinement_correction!` performs exactly one requested factor solve for a
  caller-provided residual. It does not select a tolerance, loop, change
  precision, refactor, accept a result, or choose fallback. `refine_once!`
  retains its one-step residual/error reporting wrapper. Once numerical work
  begins, correction storage may be overwritten on solve failure; no rollback
  or automatic retry is promised.
- `cholesky!(...; check=true)` throws `PosDefException` (or `DomainError` for
  non-finite input); `check=false` returns a factor with a nonzero `info`;
  `try_cholesky!` returns `nothing` on failure.
- Partial-pivoting LU accepts finite square matrices only. Singular factors
  report `FactorStatus(:singular, pivot)` with `check=false`; failed in-place
  factorization may be partially overwritten and does not roll back or trigger
  another factorization.
- `KernelConfig.thread_count` is at least one. Implemented block sizes are
  non-negative, with zero selecting the unblocked kernel. Unsupported tuning
  fields are rejected instead of being retained as inert controls.

## Factor metadata

- Factor metadata accessors for pivots, permutations, and block structures
  return defensive copies so callers cannot corrupt solve metadata.
- `factor_matrix(F)` intentionally exposes borrowed/owned numerical storage.
  Mutating it invalidates the numerical factor. Use-boundary checks detect
  precision and non-finite corruption but cannot certify arbitrary numeric
  modifications.
- Structural metadata recorded at factorization time remains cheap to query.
  Numerical summaries derived from `factor_matrix(F)` are recomputed by the
  checked diagnostics API rather than cached, because the public borrowed
  storage is mutable and a cache could become stale.
- Symmetric factor storage keeps every matrix slot as an independently owned
  MPFR value, including mirrored LDLT entries after pivoting and updates.
- In-place Cholesky rejects shared `BigFloat` objects within its authoritative
  triangle before numerical mutation. Sharing confined to the inactive
  triangle is ignored; allocating Cholesky repairs source sharing through its
  ownership-safe deep copy.
- In-place LDLT rejects authoritative lower entries that share a `BigFloat`
  object before mutation. Sharing confined to the non-authoritative upper
  triangle is harmless because that triangle is rebuilt. Allocating LDLT
  breaks source sharing through its ownership-safe deep copy.
- Every factor exposes the common public metadata protocol:
  `factor_matrix`, `factor_backend`, `factor_precision`, `factor_status`,
  `factor_kind`, `factor_triangle`, `factor_failure_position`,
  `factor_diagnostics`, and `issuccess`. Consumers do not need concrete-field
  access. Factor-specific permutation, pivot, block, and rank accessors return
  defensive copies where their result is mutable.
- Diagnostics report numerical facts without policy. Cholesky diagonal ratios,
  LDLT pivot magnitudes/determinants, RRQR accepted/rejected diagonals, and LU
  row-swap metadata never trigger a fallback, precision escalation, acceptance
  threshold, or automatic refinement.
- LDLT normalized 2x2 block quality is defined as
  `abs(det(Dblock))/max(abs(d11),abs(d12),abs(d22))^2`; it is `nothing` when no
  2x2 block exists. This is a reported scale-free quantity, not a stability
  decision.
