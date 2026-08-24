# Memory Accounting

BFLA reports Julia-allocated bytes and native (MPFR / C-side) allocation
separately. `@allocated == 0` is **not** interpreted as proof that the MPFR/C
side allocated nothing.

## How to read the numbers

* **Julia allocation** (`@allocated`) counts objects the Julia GC manages. The
  zero-allocation hot-path contract is stated in Julia bytes.
* **Native allocation** is bounded via `Sys.maxrss()` deltas. Because
  `Sys.maxrss()` only grows, a *stable* RSS across repeated factor/solve cycles
  is the honest signal that no new native blocks are retained. It cannot show a
  temporary allocation that is freed, so it is a lower bound on MPFR churn.

The cache holds every owned destination across calls, so MPFR storage for the
factor matrix and the scalar accumulators is retained and reused.

## Measured hot-path profile (Native backend, warm)

Single-call allocations after warm-up, `n = 32`, `precision = 256`:

| operation              | Julia bytes |
|------------------------|-------------|
| `factorize!` (Cholesky)| 0           |
| `solve!` (Cholesky)    | 0           |
| `factorize!` (LU)      | 0           |
| `solve!` (LU)          | 0           |
| `factorize!` (LDLT)    | allocates its metadata (see below) |
| `solve!` (LDLT)        | 0           |
| `factorize!` (RRQR)    | allocates its metadata (see below) |
| `solve!` (RRQR)        | 0           |

LDL and RRQR factorization currently reuse the allocating reference kernels for
their pivot/metadata output (`perm`, `blocks`, `tau`, `jpvt`). Their *solve*
paths and the Cholesky/LU factorization/solve paths are zero-allocation. This is
a documented, reproducible limitation rather than a hidden allocation.

## MPFR allocator behavior

MPFR itself allocates internally through its registered allocator for some
operations; BFLA cannot observe or suppress that without intercepting the
allocator, which it does not do. The properties BFLA *does* guarantee are:

1. no new Julia `BigFloat` wrapper objects are created on the zero-allocation
   hot path;
2. no owned MPFR storage is newly retained across repeated cycles (RSS is
   stable);
3. every destination owns its MPFR storage independently.

To reproduce: run `benchmark/run_baseline.jl` for the allocating baseline and
the cache-path tests in `test/caches.jl` for the zero-allocation contract.
