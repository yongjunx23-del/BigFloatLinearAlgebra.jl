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

`compare_sdpx_legacy_timing.jl` performs a correctness-gated, operation-only
timing and allocation A/B of BFLA Native against the frozen SDPX legacy kernels
(GEMM, Cholesky, TRSM, and `dot`). Mutable inputs are rebuilt outside the timed
region. It reports cold/warm median/IQR/min/max, allocations, RSS, both source
commits, and the Native/Legacy ratio. Run it with:

```julia
julia --project=. benchmark/compare_sdpx_legacy_timing.jl
```

The Legacy A/B retains both ordinary Cholesky and steady-state explicit
workspace Cholesky rows. The latter is correctness-gated first, so its timed
samples intentionally measure reuse rather than first-use buffer growth.

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

`run_production_cycles.jl` is the primary end-to-end runner. It measures four
correctness-gated combinations from identical immutable fixtures:

1. SYRK -> Cholesky;
2. Cholesky -> three independent multi-RHS solves;
3. TRSM -> Gram/SYRK -> RRQR;
4. LDLT -> multi-RHS solve.

The two Cholesky-containing combinations also retain explicit-workspace rows.
Those rows reuse one precision-matched ownership-scan buffer across samples;
the ordinary rows remain the default-path baseline.

`run_standalone.jl` retains standalone GEMM, SYRK, TRSM, Cholesky, LDLT, and
RRQR results. It reports ordinary Cholesky and explicit-workspace Cholesky as
separate workloads: the latter's cold sample includes first-use identity-buffer
growth, while warm samples reuse that buffer. Mutable operands are reset before
every timed sample outside the timed region, so allocation counts describe the
operation rather than fixture construction. Deterministic random fixtures are
constructed from exact dyadic rationals at the requested BigFloat precision,
without Float64 staging. Native and Generic results must pass scaled parity,
backward-error, rank, and inertia gates before timing is reported. Both runners
report the first post-gate sample, warm median/IQR/min/max, allocated bytes,
process peak RSS, precision, size, thread count, block size, Julia/CPU
information, and source commit. The RSS value is process-wide and is not
claimed as an exact per-kernel peak. The reported cold sample
excludes package loading and may benefit from compilation performed by the
correctness gate; use a fresh process when compilation latency itself matters.

`run_block_calibration.jl` sweeps explicit block sizes and thread counts for
GEMM, SYRK, TRSM, and Cholesky. It reports behavior only and never changes
`KernelConfig` defaults. Current blocked Level-3 paths take precedence over
threaded dispatch and are single-threaded, so the runner skips redundant
`threads > 1, block > 0` combinations. Cholesky is single-threaded in this
release and is measured only in the `threads=1` rows. Per-measurement `threads`
is the effective operation/cycle thread count, while each runner's header also
records the requested Native configuration. Configure all runners with
environment variables:

```bash
BFLA_BENCH_PRECISIONS=128,256,512 \
BFLA_BENCH_SIZES=8,16,32,64,128,256 \
BFLA_BENCH_BLOCK_SIZE=16 \
BFLA_BENCH_NATIVE_THREADS=4 \
BFLA_BENCH_SAMPLES=10 \
BFLA_BENCH_WARMUP=2 \
BFLA_SOURCE_COMMIT=$(git rev-parse HEAD) \
  julia --project=. -t 4 benchmark/run_production_cycles.jl
```

The reusable `BFLAWorkspace` remains caller-managed. Cholesky accepts it only
for the measured authoritative-triangle identity scan; the standalone runner
preserves an ordinary Cholesky row beside it so allocation and timing effects
remain auditable. No other public kernel accepts a workspace parameter.

Configuration (operation set, precisions, sizes) lives at the top of
`run_kernels.jl`. Keep sizes below `512` unless the machine has ample memory;
MPFR arithmetic at 512 bits on large matrices is memory-bandwidth bound.
