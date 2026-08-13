# Precision

BFLA operates on `BigFloat` at an explicit bit precision. The precision is
traced from the inputs and carried into every scratch object and result.

- Storage allocation requires an explicit `precision_bits` (or derives it from
  the source array's first element for `owned_similar`/`owned_copy`).
- Every array entering a public numerical kernel is scanned for a uniform
  element precision. An intra-array or cross-operand mismatch fails closed with
  `PrecisionMismatch`; validation happens once at the public boundary, so the
  internal kernels never re-scan.
- `NativeBackend` rounds each MPFR operation into a destination object created
  at the target precision, so it does not depend on the ambient global
  precision and cannot leak a low-precision scratch across 128/256/512-bit
  contexts.
- An empty `BigFloat` array has no precision to infer, so `norminf` on an empty
  array fails closed with an `ArgumentError` rather than inheriting ambient
  `setprecision`.

The reference oracle in the test suite evaluates each operation at `2p` bits
and rounds back to `p`, then compares with a scaled relative error against
`p`-bit `eps`.
