# SDPX Provider Contract

BFLA is a solver-independent dense `BigFloat`/MPFR linear-algebra backend. It is
used by SDPX-style solvers as a factor/solve *provider*. This page is the
provider contract: the stable facts BFLA exposes and the guarantees it makes. It
deliberately does **not** include SDP/IPM formulation, KKT systems, precision
escalation policy, stopping criteria, or solver fallback — those belong to the
solver, never to the backend.

## Solver-facing API

The solver consumes factors and caches through stable accessors:

* `factor_status(cache_or_factor) -> FactorStatus` — `kind` in
  `:success / :not_positive_definite / :nonfinite / :singular /
  :pivot_failure` plus an optional 1-based failure position.
* `factor_kind`, `factor_precision`, `factor_backend` — the numerical facts.
* `factor_perm`, `factor_blocks`, `factor_inertia`, `factor_rank`, `factor_jpvt`
  — permutation, pivot-block layout, Sylvester inertia, RRQR rank, and column
  permutation.
* `factor_diagnostics` — a machine-readable NamedTuple of the above.

These are **facts**, not decisions. BFLA reports a factorization that
encountered a zero pivot as `:singular`; it does not choose whether the solver
should accept it, fall back, or iterate.

## What the provider guarantees

1. **Explicit precision.** Precision comes from the operand, a `BFLAWorkspace`,
   or a `FactorCache`; never from ambient `setprecision`.
2. **Ownership safety.** Every mutable `BigFloat` destination owns independent
   MPFR storage; no shared `zeros`/`fill` objects leak into kernels.
3. **No silent fallback.** `NativeBackend` never converts to `Float64` and never
   delegates to `GenericBackend`; an unsupported operation is a hard
   `UnsupportedOperation`.
4. **Solver-independence.** BFLA reports residuals and normwise backward errors
   but does not accept/reject them, set a tolerance, choose refinement steps, or
   switch backend.
5. **Reusable storage.** Repeated `factorize!`/`solve!` through a cache reuse
   owned destinations; precision or size changes only through explicit
   `prepare!`.

## The contract in code

```julia
cache = BFLACholeskyCache(NativeBackend())
prepare!(cache, n, precision_bits)
factorize!(cache, A)
if factor_status(cache).kind !== :success
    # the solver decides what to do with the failure
    return solver_handling(factor_status(cache))
end
# x is owned by the solver (owned_zeros) -> the trusted, zero-allocation solve
solve_trusted!(x, cache, b)
eta = normwise_backward_error(NativeBackend(), A, x, b,
                              owned_zeros(BigFloat, size(b)...; precision_bits = precision_bits))
# eta is a fact; whether it is "good enough" is the solver's call.
```

The same contract holds for the ordinary allocating factors and for the
LinearSolve.jl integration, which is a thin adapter on top of the reusable
caches and introduces no hidden backend fallback or ambient precision.
