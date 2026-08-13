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

Configuration (operation set, precisions, sizes) lives at the top of
`run_kernels.jl`. Keep sizes below `512` unless the machine has ample memory;
MPFR arithmetic at 512 bits on large matrices is memory-bandwidth bound.
