# Merge Review — `refactor/owned-factor-cache` → `main` (0.2.0)

*Status: evidence for PR #5. Objective summary of the final tested state; not a
living source-of-truth contract — see the CHANGELOG and `docs/src/` for the
maintained contract.*

## Scope

Adds a public, precision-specific reusable factor-cache API and rebuilds the
LinearSolve.jl adapter on it. This review records what was implemented, tested,
and measured at merge time.

## Verified behaviour

- **Checked vs trusted solve.** `solve!(x, cache, b)` re-owns the destination
  (safe for shared/ambient-precision arrays, allocates by design).
  `solve_trusted!(x, cache, b)` is the zero-allocation hot path and requires an
  independently-owned destination at the factor precision; it rejects aliasing
  against the factor and the RHS.
- **Zero-allocation (Native, warm):** Cholesky/LU `factorize!`, `solve_trusted!`,
  and the full factorize+solve cycle are **0 Julia bytes** at 128/256/512-bit and
  sizes 8/32/128, with stable `Sys.maxrss()` (0 delta) across repeated cycles.
- **Refinement is allocation-light, NOT zero-allocation.** `refine_once!` still
  calls the generic residual/norm/axpy paths and builds fresh `BigFloat`
  constants/scalars (~1–3 KB for a size-32 vector at 256 bit). A bounded
  allocation gate and storage-identity test record this honestly.
- **LDLT/RRQR** trusted `solve_trusted!` is zero-allocation; their `factorize!`
  still allocate pivot/`tau` metadata (documented, not hidden).
- **RRQR cache is square-only (`n×n`);** rectangular inputs use the allocating
  `qr!`.

## Test results (local, Julia 1.12.6, 1 thread)

- Full `Pkg.test`: **all pass** (~6,763 tests), including cache lifecycle,
  ownership/aliasing, precision-mismatch, backend dispatch, pivot-heavy
  permutation, zero-allocation gates, shared-destination safety, metadata
  staleness-after-invalidate/failure, refinement allocation, and Native/Generic
  cross-checks, plus the rewritten LinearSolve extension suite (129 tests).
- CI matrix targets Julia 1.10/1.11/1.12 × Linux/macOS × 1/4 threads, plus a
  LinearSolve/SciMLBase lower+current compat job and a docs-build job.

## Known limitations (intentional, documented)

- LDLT/RRQR `factorize!` allocate metadata; their trusted solves are 0-alloc.
- `BFLARRQRCache` square-only.
- Cache refinement is factor-precision-only and allocation-light (no
  higher-precision residual; no zero-alloc claim).
- Cache solve scratch is single-worker (borrowed, caller-managed threading);
  within one LinearCache lifetime the RHS container shape/column count is fixed
  (a shape change requires `reinit!`/re-init).

## API surface added (0.2.0)

`AbstractFactorCache`, `BFLACholeskyCache`, `BFLALUCache`, `BFLALDLTCache`,
`BFLARRQRCache`, `prepare!`, `prepare_refinement!`, `factorize!`,
`solve!` (checked), `solve_trusted!`, `solve` (allocating), `refine_once!`,
`invalidate!`, `factor_status/precision/prepared/size`,
`factor_perm/blocks/inertia/rank/jpvt/diagnostics`.
