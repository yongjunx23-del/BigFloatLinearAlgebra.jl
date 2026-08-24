# Final Review: `refactor/owned-factor-cache` → `main`

## Decision

**Merge recommended (squash).** All P0 blockers are resolved; the branch is
numerically correct, ownership-safe, and self-consistent. No P0 correctness,
ownership, backend-identity, or silent-no-op issue remains.

## P0 fixes delivered

1. **LU cache backend dispatch** — `_cache_lu!` is dispatched on the backend
   type. `BFLALUCache{…,NativeBackend}` runs the Native kernel; a
   `BFLALUCache{…,GenericBackend}` runs the reference `_lu!(GenericBackend, …)`.
   `factor_backend(cache)` reflects the real path; no implicit fallback.
   Verified with a pivot-heavy matrix and a Generic/Native distinguishing test.
2. **Final LU permutation** — `cache.perm` is rebuilt in place from step pivots
   after every `factorize!`. `factor_perm`, `factor_diagnostics(...).permutation`
   and the LinearSolve diagnostics all return the correct final permutation,
   matching the allocating LU on a pivot-heavy input.
3. **Refinement lifecycle/API** — `invalidate!` no longer clears reusable
   refinement scratch; `prepare!(; nrhs)` honors `nrhs` via a new
   `prepare_refinement!`; the silent `residual_precision` no-op was removed
   (cache refinement is documented factor-precision-only); new tests cover warm
   allocation, object identity, and invalidate/refactor/refine with backward
   error non-worsening.
4. **RRQR public scope** — decided and documented: `BFLARRQRCache` is currently
   square-only (`n×n`); it is not presented as full rectangular RRQR.
   Rectangular inputs use the allocating `qr!`.
5. **No factor deep-copy on LinearSolve refresh** — the adapter is rebuilt
   directly on the reusable factor cache. A matrix refresh re-factorizes into
   the *same* owned storage; RHS-only solves reuse the factor; the factor-matrix
   object identity is preserved across refresh; failure never builds a
   compatibility snapshot; the allocating factor API is untouched.
6. **LinearSolve solution ownership** — the single `u_repaired::Bool` was
   replaced by a state that re-verifies solution array identity, shape, and
   precision and re-owns `cache.u` on any change (vector→matrix, column change,
   128→256 bit, replaced storage, shared `fill!`). Tests cover these cases.
7. **Ownership-safe public solve** — added `solve_trusted!(x, cache, b)`
   (trusted zero-alloc hot path requiring an owned destination) and made
   `solve!(x, cache, b)` a checked API that re-owns shared/ambient-precision
   destinations safely (no silent corruption). Shared-destination test added.

## Remaining (documented) limitations

- **LDLT / RRQR cache `factorize!`** still allocate their pivot/`tau` metadata
  (their trusted solve and refinement paths are zero-allocation). Reported
  honestly in `docs/src/memory_accounting.md`.
- **`BFLARRQRCache` is square-only.** Full rectangular RRQR remains on the
  allocating `qr!`.
- **Higher-precision residual refinement** is not part of the cache contract
  (the allocating-factor `refine_once!` retains it); the cache `refine_once!` is
  factor-precision-only.
- Cache solve scratch is single-worker (borrowed, caller-managed threading).

## Test results

- Full `Pkg.test`: **all pass** (6,738 tests), including new cache lifecycle,
  ownership, precision-mismatch, backend-dispatch, pivot-heavy-permutation,
  zero-allocation, shared-destination, refinement, and Native/Generic
  cross-check tests, plus the rewritten LinearSolve extension suite (123 tests).
- CI matrix: Julia 1.10/1.11/1.12 × Linux/macOS × 1/4 threads.

## Benchmark (Native, warm)

- Cholesky/LU cached `factorize!`, `solve_trusted!`, and full
  factorize+solve cycles: **0 Julia bytes** at 128/256/512 bit and sizes
  8/32/128, with stable `Sys.maxrss()` (0 delta) across repeated cycles.
- LDLT/RRQR `solve_trusted!`: 0 bytes; their `factorize!` allocations are
  documented, not masked.

## API changes (new public surface)

`AbstractFactorCache`, `BFLACholeskyCache`, `BFLALUCache`, `BFLALDLTCache`,
`BFLARRQRCache`, `prepare!`, `prepare_refinement!`, `factorize!`,
`solve!` (checked), `solve_trusted!`, `refine_once!`, `invalidate!`,
`factor_status`, `factor_precision`, `factor_prepared`, `factor_size`,
`factor_perm`, `factor_blocks`, `factor_inertia`, `factor_rank`, `factor_jpvt`,
`factor_diagnostics`.

## Recommendation

Squash-merge the branch into `main` and release as **0.2.0** (new public API).
CI is green; no correctness, ownership, backend-identity, or silent-no-op issue
blocks the merge.
