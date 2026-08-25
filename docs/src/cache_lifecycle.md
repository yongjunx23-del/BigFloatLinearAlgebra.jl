# Reusable Factor Caches

The reusable, precision-specific factor caches decouple *reservation* of
numeric storage from its *use*. `prepare!` and `prepare_refinement!` are the
allocation entry points; after warm-up the trusted hot path (`factorize!` for
Native Cholesky/LU and `solve_trusted!` for all four caches) writes into the
cache's own `BigFloat` destinations without creating new Julia objects. The
checked `solve!` re-owns its destination (allocating by design), and refinement
is allocation-light rather than zero-allocation.

## Ownership model

Every mutable `BigFloat` destination a cache uses is owned by the cache:

* the factor matrix (`factors`) is an independently-owned `n×n` array at an
  explicit precision;
* `CacheScalars` holds the precision-specific `0`, `1`, `-1` constants and the
  working accumulators used by the factorization and solve kernels;
* the metadata arrays (`pivots`, `perm`, `blocks`, `subdiag_is_d`, `tau`,
  `jpvt`) and the per-worker `BFLAWorkspace` solve scratch are owned by the
  cache.

Precision is never read from ambient state. The cache precision is set by
`prepare!` and every operand must match it exactly; a cross-operand precision
mismatch throws [`PrecisionMismatch`](@ref) before any storage is written.

## Lifecycle

```julia
cache = BFLACholeskyCache(NativeBackend())   # un-prepared
prepare!(cache, n, precision_bits; nrhs = 1) # reserve storage (allocating)
prepare_refinement!(cache, nrhs)              # optionally size refinement scratch
factorize!(cache, A)                          # factor A into cache.factors
solve!(x, cache, b)                           # checked: re-owns x, safe for any dest
solve_trusted!(x, cache, b)                   # trusted: zero-alloc, x must be owned
refine_once!(cache, A, x, b)                  # exactly one refinement step
factor_status(cache)                          # success / failure + position
factor_precision(cache)                       # the reserved precision
invalidate!(cache)                            # drop the factorization, keep storage
```

The cache lifecycle is *borrowed*: the caller owns the cache's lifetime, its
worker/threading policy, and its synchronization. The ordinary allocating
factor API (`cholesky`, `lu`, `ldlt`, `qr`) is untouched and remains
independent.

## Checked vs trusted solve

* `solve!(x, cache, b)` is the **checked** API. It validates status, dimensions,
  RHS precision, and aliasing, and *re-owns* the destination (replacing each
  element with an independently-owned copy at the factor precision). This makes
  it safe for any destination, including a shared `fill!` array or one carrying
  a stale ambient precision, at the cost of allocating the copies.
* `solve_trusted!(x, cache, b)` is the **solver-facing hot API**. It requires
  `x` to already be independently owned at the factor precision (for example
  from `owned_zeros`), copies `b` into the existing `x` objects and solves in
  place, allocating no new `BigFloat` objects. SDPX should call this on the hot
  path and guarantee `x`'s ownership.

## Factor-integrity contract

The checked paths (`solve!`, `refine_once!`, `factor_diagnostics`,
`factor_inertia`, `numerical_rank`, and the metadata accessors that read factor
internals) re-validate the owned factor through a single entry,
`_validate_factor_integrity!`, which checks, in order: factor **shape** (square
for Cholesky/LU/LDLT, `m×n` for RRQR, and `size(cache.factors) == (cache.n,
cache.n)` for a cache), factor storage **precision**, factor storage
**finiteness**, and **metadata** consistency. This mirrors the ordinary
allocating factor API where `ldiv!` rescans the factor while `ldiv_trusted!`
skips it. The trusted paths (`solve_trusted!`, `refine_once_trusted!`) skip the
factor rescan — their caller guarantees the factor is unchanged — but still
reject a non-finite RHS and a non-finite solve result. A caller that mutates
`factor_matrix(cache)` must `invalidate!`/re-`factorize!` before a checked use;
the checked paths throw a clear error (never a `BoundsError`, segfault, or
silent wrong result) on malformed metadata such as an out-of-range pivot, an
inconsistent permutation, invalid LDLT pivot blocks, an invalid RRQR
rank/`tau`/`jpvt`, or an in-range-but-wrong RRQR rank/threshold/scale/tolerance
that a pure range check would accept.

The allocating `solve(cache, b)` also goes through the checked entry: it
validates factor shape/storage/metadata and RHS finiteness before solving, so a
caller that mutated the cache's factor or metadata fails closed instead of
silently producing a wrong result.

## Invariants

1. **Precision or size change is explicit.** Changing `n` or the precision is a
   new `prepare!`; it never happens inside a hot loop.
2. **No element replacement on the trusted hot path.** `factorize!` and
   `solve_trusted!` write into the cache's existing `BigFloat` objects. (The
   checked `solve!` intentionally re-owns.) Refinement writes into its owned
   scratch but still builds a few fresh scalars — it is allocation-light, not
   zero-allocation.
3. **Metadata reuse scope.** Cholesky and LU Native `factorize!` write
   `pivots`/`perm` into the cache's preallocated arrays (no wrapper or identity
   vector creation). LDLT and RRQR `factorize!` currently replace their
   `perm`/`blocks`/`tau`/`jpvt` arrays with freshly-allocated ones, so their
   factorization is *not* zero-allocation; their trusted solves are.
4. **Zero Julia allocation after warm-up.** For the Native backend,
   `factorize!` + `solve_trusted!` for Cholesky and LU, and every cached trusted
   solve, allocate 0 Julia bytes once the destination storage
   carries the cache precision and is independently owned.
5. **No implicit `Float64` conversion and no silent fallback.** The Native
   backend never converts to `Float64` and never delegates to the Generic
   backend.

## Warm-up ownership repair

A freshly created solution array (for example the array LinearSolve.jl builds
with `similar`+`fill!`) can carry a stale, ambient precision and may install the
*same* `BigFloat` object in every slot. Before such storage is reused the cache
re-owns it once (replacing each element with an independent object at the
requested precision). This one-time cost is the documented warm-up; it is not
part of the repeated numeric hot path, which writes into the already-owned
destinations.
