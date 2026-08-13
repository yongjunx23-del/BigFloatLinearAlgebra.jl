# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - unreleased

### Added

- `AbstractBFLABackend`, `NativeBackend`, and `GenericBackend` backend types.
- `capabilities(backend)` audit hook returning a fixed capability tuple.
- Ownership-safe storage API: `owned_zeros`, `owned_similar`, `owned_copy`,
  `copy_owned!`, `zero_owned!`, and `fill_owned!`.
- BLAS Level 1: `scal!`, `axpy!`, `axpby!`, `dot`, and `norminf`.
- BLAS Level 2: `gemv!`, `trsv!`, and `syr!`.
- BLAS Level 3: `gemm!`, `syrk!`, `trmm!`, and `trsm!`.
- `mirror_triangle!` for symmetric triangle completion.
- Lower-triangular Cholesky factorization and solve:
  `try_cholesky!`, `cholesky!`, `cholesky`, `ldiv!`, `solve!`, and `solve`.
- `NativeBackend` kernels extracted and generalized from the SDPX legacy
  BigFloat dense kernels (see `THIRD_PARTY_NOTICES.md`).
- `GenericBackend` reference implementations built on `LinearAlgebra` generic
  methods.

### Reference

The Native backend derives from SDPX `src/kernels/bigfloat.jl` as found at the
time of extraction. No SDPX source is vendored; attribution is retained in
`THIRD_PARTY_NOTICES.md`.

### Performance

The `NativeBackend` GEMM and SYRK kernels hoist `TransposeOp` flags to
compile-time `Val` parameters, eliminating the O(n^2) heap allocation previously
caused by runtime `Val` dispatch boxing mutable MPFR scratch. GEMM/SYRK now
allocate a constant number of bytes (matching the SDPX legacy owned kernels),
and Cholesky, TRSM, and `dot` are at allocation parity or better than the
frozen legacy path.
