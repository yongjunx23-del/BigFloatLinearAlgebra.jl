# Benchmarks

The benchmark runner is intentionally separate from the unit test suite so
timing noise never gates CI. Run it from the package root:

```julia
julia --project=. -t 4 benchmark/run_kernels.jl
```

`run_kernels.jl` performs, for every configured operation/precision/size:

1. a correctness gate (Native vs. Generic backward error);
2. at least 2 warmup samples and 10 timed samples;
3. reporting of median, IQR, min, max, and allocated bytes.

`compare_sdpx_legacy.jl` is an opt-in integration benchmark that loads the
frozen SDPX legacy BigFloat kernels by path and compares them against the BFLA
Native backend. It is not part of BFLA's required test dependencies.

`compare_sdpx_legacy_timing.jl` performs a correctness-gated timing and
allocation A/B of BFLA Native against the frozen SDPX legacy kernels (GEMM,
Cholesky, TRSM, and `dot`), reporting median/IQR/min/max and allocations. Run it
with:

```julia
julia --project=. benchmark/compare_sdpx_legacy_timing.jl
```

`precision_scan_overhead.jl` quantifies the cost of the strict uniform-precision
scan at the public boundary versus the raw kernel, for both O(n) (`dot`) and
O(n²) (`gemm`) operations. It exists to detect whether validation ever becomes
a material fraction of repeated factor-solve cycles.

`run_dense_cycle.jl` measures the complete solver-like dense path added through
Phases 1-11: SYRK assembly, Cholesky, two-RHS solve, explicit q-bit residual,
one refinement correction, and total cycle. It correctness-gates both Native
and Generic from identical immutable fixtures and reports absolute
median/IQR/min/max, median bytes, backward error, source commit, and RSS.

```bash
BFLA_SOURCE_COMMIT=$(git rev-parse HEAD) \
  julia --project=. benchmark/run_dense_cycle.jl
```

Defaults are 128/256/512 bits, sizes 16/32/64, 2 warmups, and 10 samples.
Comma-separated `BFLA_BENCH_PRECISIONS`/`BFLA_BENCH_SIZES` and integer
`BFLA_BENCH_NATIVE_THREADS` may define a reproducible smaller or larger run.

Configuration (operation set, precisions, sizes) lives at the top of
`run_kernels.jl`. Keep sizes below `512` unless the machine has ample memory;
MPFR arithmetic at 512 bits on large matrices is memory-bandwidth bound.
