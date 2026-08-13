# Third-party notices

The `NativeBackend` kernels in `src/native_backend.jl` and `src/mpfr.jl` are
extracted and generalized from the BigFloat dense kernels in
`SDPX.jl` (`src/kernels/bigfloat.jl`, commit lineage documented in
`CHANGELOG.md`).

SDPX.jl is distributed under the MIT License:

    Copyright (c) 2022 Li-Yuan Chiang (SDPJSolver.jl, from which SDPX.jl is derived)
    Copyright (c) 2026 SDPX.jl contributors

The extraction preserves the original reduction order, MPFR ownership
discipline, and direct `mpfr_*` rounding calls so that the BFLA Native backend
stays numerically equivalent to the frozen SDPX legacy path. Solver-specific
concepts (KKT, Schur, cone, certificate, regularization, line-search helpers)
were removed during the extraction.

`MutableArithmetics.jl` is used by the Native backend for allocation-free,
ownership-preserving MPFR accumulation. It is distributed under the Mozilla
Public License 2.0 (see https://mozilla.org/MPL/2.0/).
