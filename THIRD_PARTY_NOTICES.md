# Third-party notices

The `NativeBackend` kernels in `src/native_backend.jl` and `src/mpfr.jl` are
extracted and generalized from the BigFloat dense kernels in
`SDPX.jl`.

Extraction provenance:

- source repository: <https://github.com/yongjunx23-del/SDPX.jl>
- source path: `src/kernels/bigfloat.jl`
- source commit SHA:
  `d6b2198102720e509704cb66c124432923a0628f`
  ("Revert unstable BigFloat reciprocal reuse")
- source blob SHA-1: `318bb4c32998ecf8ef56d4ff25be02a5717ad509`

The frozen parity environment used during the 2026-08-13 review was at SDPX
commit `b0c4eaa7e4ee306983dbfabf65d0d005a11ed220`; its copy of the source path has
the same blob SHA-1. No SDPX source is vendored and SDPX is not a package
dependency.

SDPX.jl is distributed under the MIT License:

    Copyright (c) 2022 Li-Yuan Chiang (SDPJSolver.jl, from which SDPX.jl is derived)
    Copyright (c) 2026 SDPX.jl contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The extraction preserves the original reduction order, MPFR ownership
discipline, and direct `mpfr_*` rounding calls so that the BFLA Native backend
stays numerically equivalent to the frozen SDPX legacy path. Solver-specific
concepts (KKT, Schur, cone, certificate, regularization, line-search helpers)
were removed during the extraction.

`MutableArithmetics.jl` is used by the Native backend for allocation-free,
ownership-preserving MPFR accumulation. It is distributed under the Mozilla
Public License 2.0 (see https://mozilla.org/MPL/2.0/).
