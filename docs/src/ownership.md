# Ownership

Julia's `BigFloat` is a mutable MPFR object, so array construction does not
guarantee element independence:

- `zeros(BigFloat, m, n)` and `fill(BigFloat(...), n)` install one shared object
  in every slot.
- `copy` and `Matrix(A)` copy object references, not values.

Mutating a slot in such an array corrupts every slot that shares the object.
BFLA's storage API guarantees independence:

- `owned_zeros(BigFloat, dims...; precision_bits)` returns a zero array whose
  elements are independent MPFR objects at exactly `precision_bits`.
- `owned_similar(A; precision_bits)` allocates owned zero storage with `A`'s
  shape.
- `owned_copy(A; precision_bits)` is a deep numeric copy with independent
  storage.
- `copy_owned!`, `zero_owned!`, and `fill_owned!` install independent MPFR
  objects, so they are safe even for aliased input arrays.

In-place BLAS kernels require independently owned destination storage. Pass
storage created by `owned_zeros`/`owned_similar`/`owned_copy`. A destructive
ownership probe (which the library deliberately does not expose as a public,
reliable predicate) lives in the test tooling.

## In-place vs. allocating factorization

- `cholesky!(backend, A)` borrows `A` and overwrites its authoritative triangle.
- `cholesky(backend, A)` deep-copies `A` first, so the factor owns its storage
  and the input is untouched.
- `solve!(factor, rhs)` overwrites `rhs`; `solve(factor, rhs)` copies `rhs`
  first.
