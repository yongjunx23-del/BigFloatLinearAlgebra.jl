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
- `convert_owned!(destination, source)` is the explicit reusable
  cross-precision path. Destination precision is authoritative, source
  elements may carry different precisions, and values are rounded into the
  existing destination objects so destination ownership and object identity
  are preserved. Array overlap and cross-array MPFR object sharing are
  rejected.
- `copy_owned!`, `zero_owned!`, and `fill_owned!` install independent MPFR
  objects. `copy_owned!` rejects overlapping array storage but safely breaks
  shallow element sharing between otherwise distinct arrays. `fill_owned!` is
  precision-strict and validates a uniform destination plus a same-precision
  value before modifying any slot.

In-place BLAS kernels require independently owned destination storage. Pass
storage created by `owned_zeros`/`owned_similar`/`owned_copy`. A destructive
ownership probe (which the library deliberately does not expose as a public,
reliable predicate) lives in the test tooling.

In-place LDLT validates independent ownership across its authoritative lower
triangle before mutation. It rejects shared lower entries with `ArgumentError`;
sharing confined to the stale upper triangle is allowed because that triangle
is rebuilt with independent numeric copies. Allocating `ldlt` deep-copies and
therefore repairs a pre-aliased source.

## In-place vs. allocating factorization

- `cholesky!(backend, A)` borrows `A` and overwrites its authoritative triangle.
- `cholesky(backend, A)` deep-copies `A` first, so the factor owns its storage
  and the input is untouched.
- `solve!(factor, rhs)` overwrites `rhs`; `solve(factor, rhs)` copies `rhs`
  first.
