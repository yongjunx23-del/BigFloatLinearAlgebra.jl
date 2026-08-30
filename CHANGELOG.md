# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-08-30

### Added

- Optional QDLDL sparse signed-LDL cache integration, including explicit
  provider availability, frozen-pattern, factor-state, and authority-failure
  handling.  Unsupported sparse QDLDL configurations fail closed.

### Fixed

- Optional QDLDL test dependencies are isolated so the base package and its
  ordinary factor-cache contract remain independently runnable.

## [0.2.2] - 2026-08-25

### Fixed

- A successful Cholesky factor with a non-positive diagonal element (mutated
  while keeping shape, precision, and finiteness intact) is now rejected by the
  checked `solve!`/`ldiv!`, the cache checked `solve!`, the allocating
  `solve(cache, b)`, and `factor_diagnostics`, instead of silently producing a
  wrong result. The canonical positive-diagonal check lives in
  `_validate_cholesky_metadata` and is reached only for a `:success` factor.
- The cache trusted `solve_trusted!` now enforces the factor-precision contract
  on both the solution and the RHS, mirroring the ordinary `ldiv_trusted!`
  behavior: a solution or RHS whose precision differs from the cache's factor
  precision throws `PrecisionMismatch`. The check reuses the existing precision
  scan, so the zero-Julia-allocation trusted gate is preserved.

## [0.2.1] - 2026-08-25

### Added

- A single checked factor-integrity entry, `_validate_factor_integrity!`, that
  validates factor **shape** (square for Cholesky/LU/LDLT, `m×n` for RRQR, and
  `size(cache.factors) == (cache.n, cache.n)` for a cache), factor storage
  **precision**, factor storage **finiteness**, and **metadata** consistency. It
  is wired into the ordinary checked `ldiv!`/`solve!`, the cache checked
  `solve!`/`refine_once!`, `factor_diagnostics`, `factor_inertia`,
  `numerical_rank`, and every metadata accessor that reads factor internals.
  The trusted paths (`ldiv_trusted!`, `solve_trusted!`, `refine_once_trusted!`)
  still skip the factor rescan.
- The allocating `solve(cache, b)` now goes through a checked
  `_solve_owned_checked!` entry that validates factor shape/storage/metadata and
  RHS finiteness before solving, instead of calling `solve_trusted!` and
  skipping factor-integrity validation.
- RRQR rank-policy **semantic consistency**: the checked validator now rejects
  in-range-but-wrong corruption (rank, effective threshold, reference scale,
  tolerance-vs-atol, or a rank-policy scalar at the wrong precision) that a pure
  range check would accept. `BFLARRQRCache.factorize!` also preflights the
  precision of `A`, `tol`, `atol`, and `rtol` against the cache precision before
  mutating any cache field.
- A unified `_validate_cache_prepare` preflight for every cache's `prepare!`
  (non-negative `n`, positive `precision_bits`, `nrhs >= 1`,
  `workspace_workers >= 1`), run before any cache field is mutated so a failed
  `prepare!` leaves the cache unchanged.
- Adversarial shape and semantic fuzz suites with a residual/backward-error
  gate: a checked operation that succeeds must produce a small residual, never a
  silent wrong result.

### Changed

- Metadata accessors that read factor internals (`factor_perm`, `factor_pivots`,
  `factor_blocks`, `factor_inertia`, `factor_rank`, `factor_jpvt`,
  `factor_Rdiag`, `factor_rank_atol/rtol/scale/threshold`, `factor_diagnostics`,
  `numerical_rank`) now validate the factor before returning a value, so a caller
  cannot get a plausible-looking value from a factor whose metadata was
  externally mutated. Status/backend/precision/kind accessors need no
  validation.
- Corrected stale documentation: `prepare!` is no longer described as "the only
  allocating entry point" (`prepare_refinement!` is also one), and the
  `BigFloatLU`/`BigFloatCholesky` docstrings no longer claim the factor owns a
  "deep matrix copy".

## [0.2.0] - 2026-08-25

### Added

- New public, precision-specific reusable factor-cache API:
  `AbstractFactorCache`, `BFLACholeskyCache`, `BFLALUCache`, `BFLALDLTCache`,
  and `BFLARRQRCache`, with `prepare!`, `factorize!`, `solve!`,
  `solve_trusted!`, `refine_once!`, `prepare_refinement!`, `invalidate!`,
  `factor_status`, `factor_precision`, `factor_prepared`, `factor_size`, and
  metadata accessors (`factor_perm`, `factor_blocks`, `factor_inertia`,
  `factor_rank`, `factor_jpvt`, `factor_diagnostics`).
- A checked `solve!(x, cache, b)` (re-owns a shared/ambient-precision
  destination safely) and a solver-facing `solve_trusted!(x, cache, b)` that is
  the zero-allocation hot path for an already-owned destination.
- Zero-Julia-allocation Native Cholesky and LU `factorize!` + `solve_trusted!` full-cycle
  hot paths after warm-up (verified across 128/256/512 bit and sizes 8/32/128).
- LinearSolve.jl adapter rebuilt directly on the reusable factor cache: an
  RHS-only solve allocates no new `BigFloat` element and a matrix refresh
  re-factorizes into the same owned storage (no factor deep-copy). The adapter
  re-verifies solution shape, precision, and array identity on every solve.
- New documentation: factor-cache lifecycle, memory accounting (Julia vs native
  allocation), and the SDPX provider contract.

### Changed

- LU cache factorization is dispatched on the backend type: a `GenericBackend`
  cache executes the reference `_lu!` path, a `NativeBackend` cache the Native
  kernel; `factor_backend(cache)` reflects the real execution path and there is
  no implicit fallback.
- `cache.perm` is rebuilt from step pivots after every LU `factorize!`, so
  `factor_perm`/`factor_diagnostics` report the correct final permutation.
- `prepare!(; nrhs)` now honors `nrhs` by eagerly allocating refinement scratch
  via `prepare_refinement!`; `invalidate!` preserves reusable refinement
  storage; the cache `refine_once!` no longer accepts a silently-ignored
  `residual_precision` keyword (cache refinement is factor-precision-only).
- Unified factor-integrity validation (`_validate_factor_shape` /
  `_validate_factor_metadata`) shared by the ordinary allocating factors and the
  reusable caches: LU pivot step range + permutation consistency, LDLT
  perm/blocks/subdiag consistency, RRQR tau length/precision + jpvt permutation
  + rank + rank-policy scalars. Wired into the checked `ldiv!`/`solve!`/
  `factor_diagnostics`/`factor_inertia` paths; the trusted paths skip the
  rescan. Malformed metadata now throws a clear error instead of a `BoundsError`
  or segfault.
- Refactor exception atomicity: cache `factorize!` runs preflight checks before
  mutating factor storage and never leaves a stale `:success` status on an
  exception; failure/recovery is explicit.
- Cache solve finite semantics: checked and trusted solves reject a non-finite
  RHS and a non-finite solve result (pure scans, no Julia allocation).
- Added `refine_once_trusted!` (solver-facing) alongside the checked
  `refine_once!`; both reject solution aliasing against the factor and the RHS,
  and the checked path re-owns a shared/ambient-precision destination safely.
- `prepare_refinement!(cache, rhs_template)` preserves the exact RHS shape
  (Vector vs `n×1` vs `n×k` Matrix); `refine_once!` throws on a prepared-scratch
  shape mismatch instead of silently resizing.
- All four caches' trusted `solve_trusted!` are now zero-Julia-allocation after
  warm-up (LDLT and RRQR included); a bounded gate records that `refine_once!`
  is allocation-light, not zero.
- Metadata accessors (`factor_pivots`, `factor_perm`, `factor_blocks`,
  `factor_inertia`, `factor_rank`, `factor_jpvt`, `factor_Rdiag`,
  `factor_rank_atol/rtol/scale/threshold`, `factor_diagnostics`) throw after
  `invalidate!`/failed factorization instead of returning stale metadata.
- LinearSolve adapter re-owns `cache.u` on a same-array shared re-fill and
  rethrows interrupts/out-of-memory (only precision errors are handled).
- `prepare_refinement!` is now a checked public API: it requires a prepared
  cache, accepts only `AbstractVecOrMat{BigFloat}` templates, and throws a clear
  error on use-before-prepare, precision, or shape mismatch (no silent resize).
- A canonical `docs/src/reference.md` API reference (via `@autodocs`) documents
  every exported binding; `makedocs(checkdocs = :exports)` enforces it.
- Malformed-factor fuzz test suite (no crash / no out-of-bounds on corrupted
  metadata).

### Known limitations

- `BFLARRQRCache` is currently square-only (`n × n`); rectangular/overdetermined
  systems use the allocating `qr!`.
- LDLT and RRQR cache `factorize!` still allocate their pivot/`tau` metadata
  (their solve paths are zero-allocation). This is reported honestly in the
  memory-accounting documentation.

## [0.1.1] - 2026-08-25

### Added

- Julia package extension integration with LinearSolve.jl and SciMLBase.jl.
- `BigFloatLU()` and `BigFloatCholesky()` LinearSolve algorithms backed by
  BFLA-owned factors and explicit Native/Generic backend selection.
- Focused extension tests covering arbitrary precision, vector and matrix
  right-hand sides, cache reuse, refactorization, ownership, and failure paths.

### Changed

- LinearSolve caches reuse fresh factorizations for repeated solves and repair
  right-hand-side storage when precision or mutable BigFloat ownership changes.
- CI tests current LinearSolve releases and the supported lower compatibility
  bounds without making LinearSolve a required runtime dependency.
- TagBot now uses a repository-scoped write deploy key to publish tags whose
  registered commits contain GitHub workflow changes.

## [0.1.0] - 2026-08-17

### Added

- `AbstractBFLABackend`, `NativeBackend`, and `GenericBackend` backend types.
- `capabilities(backend)` audit hook returning a fixed capability tuple.
- Ownership-safe storage API: `owned_zeros`, `owned_similar`, `owned_copy`,
  `copy_owned!`, `zero_owned!`, and `fill_owned!`.
- BLAS Level 1: `scal!`, `axpy!`, `axpby!`, `dot`, and `norminf`.
- BLAS Level 2: `gemv!`, `trsv!`, and `syr!`.
- BLAS Level 3: `gemm!`, `syrk!`, `trmm!`, and `trsm!`.
- `mirror_triangle!` for symmetric triangle completion.
- Lower-triangular Cholesky factorization and solve:
  `try_cholesky!`, `cholesky!`, `cholesky`, `ldiv!`, `solve!`, and `solve`.
- `NativeBackend` kernels extracted and generalized from the SDPX legacy
  BigFloat dense kernels (see `THIRD_PARTY_NOTICES.md`).
- Immutable `KernelConfig` and precision-scoped `BFLAWorkspace` storage.
- Explicit `ldiv_trusted!` repeated-solve API plus caller-owned solve workspace
  support for Cholesky, LDLT, RRQR, and LU. The checked `ldiv!` contract is
  unchanged; trusted solves retain status, shape, RHS, precision, alias,
  workspace, and backend validation while skipping the caller-guaranteed live
  factor scan.
- Correction-only `refinement_correction!`, which performs exactly one
  requested factor solve without solver tolerance, iteration, refactor,
  precision escalation, acceptance, or fallback policy.
- Blocked and explicitly threaded Native GEMM, SYRK, TRSM, and Cholesky paths.
- Solver-relevant symmetric kernels: `gemmt!`, `symv!`, and `syr2k!`.
- Symmetric-indefinite Bunch-Kaufman LDLT factors, solves, inertia, pivot
  structure, and diagnostics.
- Scale-aware LDLT 2x2 pivot factor/solve/inertia arithmetic and normalized
  Bunch-Kaufman comparisons, avoiding avoidable overflow for finite
  extreme-range blocks.
- Column-pivoted rank-revealing QR with explicit permutation, numerical rank,
  R diagonal, Q/Q-transpose application, and vector/multi-RHS solve.
- Stable RRQR column norms with guarded exact trailing-norm recomputation,
  preserving deterministic pivots and caller-owned rank tolerance policy.
- Caller-owned residual computation and normwise backward-error measurement
  for ordinary and transposed vector/multi-RHS systems.
- Explicit higher-precision residual computation with p/q precision diagnostics
  and no ambient precision dependence or automatic policy decisions.
- One-step iterative refinement through the factor's recorded backend, with
  caller-owned residual/correction storage and before/after error diagnostics.
- `convert_owned!` for explicit reusable conversion into caller-owned
  destination precision while preserving destination MPFR object identity.
- Dense square partial-pivoting LU with explicit row-swap/permutation
  diagnostics and vector/multi-RHS solves for both backends.
- `GenericBackend` reference implementations built on `LinearAlgebra` generic
  methods.

### Reference

The Native backend derives from SDPX `src/kernels/bigfloat.jl` as found at the
time of extraction. No SDPX source is vendored; attribution is retained in
`THIRD_PARTY_NOTICES.md`.

### Performance

The `NativeBackend` GEMM and SYRK kernels hoist `TransposeOp` flags to
compile-time `Val` parameters, eliminating the O(n^2) heap allocation previously
caused by runtime `Val` dispatch boxing mutable MPFR scratch. GEMM/SYRK now
allocate a constant number of bytes (matching the SDPX legacy owned kernels),
and Cholesky, TRSM, and `dot` are at allocation parity or better than the
frozen legacy path.

### Changed

- Enforce a uniform-precision invariant across every element of every
  `BigFloat` array at the public API boundary; intra-array and cross-operand
  mismatches now fail closed with `PrecisionMismatch`.
- `capabilities(backend)` now reports backend-specific `cholesky_triangles`
  (`(Lower,)` for Native, `(Lower, Upper)` for Generic) using the public
  `Triangle` enum.
- `norminf` on an empty `BigFloat` array fails closed instead of inheriting
  ambient `setprecision`.
- CI runs tests with an explicit `--threads` flag and asserts the real
  `Threads.nthreads()` against the matrix axis.
- Test harness includes `test/test_utils.jl` once, removing method-overwrite
  warnings.
- `owned_copy`/`owned_similar` now require a uniform source precision by
  default; explicit cross-precision conversion is only via `precision_bits`.
- `ldiv!`/`solve!` re-verify factor storage precision against the recorded
  factor precision instead of trusting factorization-time metadata.
- Factorization status is now a machine-readable `FactorStatus`
  (`:success`, `:not_positive_definite`, `:nonfinite`, `:singular`,
  `:pivot_failure`) with an optional failure position, replacing integer
  sentinels.
- Factor permutation, pivot, and block-structure accessors return defensive
  copies, and all BFLA factor types support `size(F, dimension)`.
- Cholesky, LDLT, QR, and LU factor-use boundaries reject non-finite
  authoritative factor storage and right-hand sides before mutation, and reject
  non-finite results instead of reporting a successful solve.
- LDLT now treats only the lower triangle as authoritative, rebuilding the
  inactive upper triangle before symmetric pivoting so stale or poisoned upper
  data cannot affect the factorization.
- Ownership conversion now distinguishes overlapping array storage from
  cross-array sharing of mutable MPFR objects: `convert_owned!` rejects both,
  while `copy_owned!` safely repairs the latter with fresh destination objects.
- Native LDLT pivot decisions now use explicit factor-precision scratch for
  absolute values and Bunch-Kaufman thresholds, independent of ambient
  `setprecision`.
- `gemmt!` and `syr2k!` now validate every transformed outer and contraction
  dimension before entering a kernel, leaving the destination unchanged on
  failure.
- LDLT mirrored entries now use ownership-preserving MPFR copies, including
  after 1x1 and 2x2 pivots, trailing updates, and symmetric permutations.
- In-place LDLT now rejects pre-aliased authoritative-lower `BigFloat` entries
  before mutation, so every successful factor satisfies the independent-storage
  invariant without weakening lower-triangle authority.
- LDLT solve, QR Q application, and QR solve now dispatch explicitly through
  the backend stored in the factor and reject unsupported backends before
  mutation.
- `fill_owned!` now validates the fill value and the complete destination
  precision before mutation, making precision failures atomic.
- `mirror_triangle!` now validates complete matrix precision before copying,
  so mixed-precision failures leave both triangles unchanged.
- `KernelConfig` now rejects non-positive thread counts and negative block
  sizes; the unused `ldlt_block` field was removed rather than exposing an
  inert tuning control.
- `BFLAWorkspace` is now documented as caller-managed scratch, and ineffective
  `workspace=` kernel keywords were removed instead of being silently ignored.

### Final review hardening

- Complete the Bunch-Kaufman pivot decision with the intermediate keep-current
  1x1 test, preventing a nonsingular matrix from being rejected after selecting
  an avoidably singular 2x2 pivot block.
- Configurable Level-3/Cholesky dispatchers preserve the standard
  `UnsupportedOperation` contract for unregistered backends, and LDLT now
  resolves backend support before rebuilding the inactive triangle.

### API and diagnostics freeze

- In-place Cholesky now rejects shared mutable `BigFloat` storage within the
  selected authoritative triangle before any numerical mutation. Inactive
  triangle sharing remains irrelevant, and allocating Cholesky repairs aliased
  source storage through an ownership-safe deep copy.
- Cholesky, LDLT, RRQR, and LU now share a public factor metadata protocol,
  including `factor_triangle` and `factor_failure_position`, so consumers do
  not need concrete factor fields.
- Backend capabilities now state explicitly that QR is column-pivoted and
  rank-revealing rather than unpivoted.
- RRQR rank selection now supports explicit absolute and relative tolerances,
  records the largest input-column norm and effective threshold, and exposes
  scale-aware `numerical_rank` re-evaluation plus defensive rank metadata.
  The legacy `tol` spelling remains an absolute-only compatibility mode.
- Public factor diagnostics now expose Cholesky diagonal range/ratio, LDLT
  minimum 1x1 pivot and 2x2 determinant/normalized quality, and RRQR accepted
  and rejected diagonal facts. These APIs validate live precision and finite
  storage and do not make solver-policy decisions.
- RRQR auxiliary metadata now participates in every factor-use precision and
  finite check, including `refine_once!` validation before residual or
  correction storage can be modified.
- Add correctness-gated production-cycle, standalone, and explicit
  block/thread benchmark runners. They report cold/warm timing, IQR,
  allocations, RSS, precision, size, threads, block size, and source commit;
  mutable operands are rebuilt outside standalone timed regions and fixtures
  use exact rational values rather than Float64 staging.
- Harden the opt-in frozen SDPX Legacy A/B so every reported operation passes
  bitwise parity and mutable setup is excluded from operation timing.
- Restore the frozen row/column access path for untransposed Native GEMM while
  preserving its buffered MPFR reduction order and public validation contract.
- Restore row-segment access in unblocked lower Native Cholesky without
  changing authoritative-triangle, failure-position, or MPFR trajectory rules.
- Benchmark output now reports effective thread counts and block calibration
  skips configurations whose thread count is ignored by blocked or Cholesky
  dispatch.
- Cholesky can explicitly reuse a precision-matched, worker-local workspace
  buffer for its authoritative-triangle ownership scan. The workspace stores
  only object identifiers, preserves the exact numerical path, and is rejected
  before matrix mutation on precision or worker mismatch.
