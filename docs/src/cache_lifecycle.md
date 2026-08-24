# Reusable Factor Caches

The reusable, precision-specific factor caches decouple *reservation* of
numeric storage from its *use*. `prepare!` is the only allocating entry point;
after warm-up the `factorize!` / `solve!` / refinement hot paths write into the
cache's own `BigFloat` destinations without creating new Julia objects.

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
