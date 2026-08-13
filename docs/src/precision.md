# Precision

BFLA operates on `BigFloat` at an explicit bit precision. The precision is
traced from the inputs and carried into every scratch object and result.

- Storage allocation requires an explicit `precision_bits` (or derives it from
  a full uniform-precision scan for `owned_similar`/`owned_copy`).
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
- Ordinary kernels, `residual!`, and `normwise_backward_error` remain strict
  same-precision operations. `higher_precision_residual!` is a separate,
  explicit exception: its inputs share factor precision `p`, the required
  `residual_precision=q` must match its caller-owned destination, and it
  requires `q > p`.
  Every multiply/add is rounded into q-bit scratch without consulting ambient
  precision, and its diagnostics report both precisions.
- `refine_once!` explicitly rounds a p- or q-bit residual into caller-owned
  p-bit correction storage before dispatching to the factor's p-bit solve. It
  performs one correction only and never changes precision automatically.
- `convert_owned!` is the reusable explicit conversion operation. Its
  destination must be uniformly precise and determines the rounding precision;
  source entries may have different precisions. `copy_owned!` remains a strict
  same-precision copy and `owned_copy(A)` remains strict unless its
  `precision_bits` keyword is supplied.
- `fill_owned!` and `mirror_triangle!` validate complete storage precision
  before mutation. Neither operation is an implicit precision-conversion path.

The reference oracle in the test suite evaluates each operation at `2p` bits
and rounds back to `p`, then compares with a scaled relative error against
`p`-bit `eps`.
