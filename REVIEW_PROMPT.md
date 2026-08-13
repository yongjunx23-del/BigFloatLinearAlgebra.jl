# BigFloatLinearAlgebra.jl independent final-review prompt

请解压附件并独立审查 `BigFloatLinearAlgebra.jl`。不要默认 README、测试或本提示词的结论正确；以实际源码、git snapshot 和可复现反例为准。请先找 correctness、ownership、precision、fallback 和数值语义问题，再看性能与文档。

## Snapshot

```text
base origin/main: 8e71456e95f5ddcbd0e803633c8ed9796ac9b70b
review branch: development/bfla-api-diagnostics-performance
expected packaged HEAD: 在 ZIP 文件名和交付消息中给出；请用源码内容核对
Julia local test version: 1.12.6
```

本轮主要提交从 base 依次为：

```text
243d37a Freeze factor protocol and RRQR rank semantics
339fd98 Unify factor diagnostics and quality metadata
0fd612e Validate RRQR metadata before refinement
fbb51ed Add correctness-gated production benchmarks
94ebd62 Harden frozen SDPX legacy benchmarks
0a3078c Restore native GEMM row-column fast path
30d69e5 Restore native Cholesky row-segment fast path
5999c32 Report effective benchmark dispatch settings
771041e Reuse workspace for Cholesky ownership scans
137e238 Benchmark workspace Cholesky against SDPX legacy
```

## Invariants

- BFLA 是独立 dense BigFloat/MPFR provider；不得依赖 SDPX 或出现 solver policy。
- 所有操作显式 backend；禁止 Native -> Generic、Cholesky -> LDLT/QR 等隐藏 fallback。
- Native 不得使用 Float64 staging 或 ambient `setprecision`。
- ordinary operations 同精度 fail closed；跨精度仅允许显式 API。
- mutable BigFloat destination/factor authoritative storage 必须独立所有。
- Cholesky/LDLT 只读取权威三角；inactive triangle 的 stale/NaN/shared storage 不影响合法路径。
- factor solve 必须通过 factor 记录的 backend；consumer 不读取 concrete private fields。
- diagnostics 只报告 numerical facts，不决定阈值、fallback、refinement loop 或 precision escalation。
- `refine_once!` 恰好一步。
- workspace 必须显式、precision-matched、worker-local；不得改变 backend/数值路径或被静默忽略。

## Review focus

1. **Cholesky workspace and ownership**
   - `_workspace_identity_buffer` 的 precision/worker 检查顺序是否在 matrix mutation 前；
   - `objectid + sort + IdDict collision check` 是否可能漏检真实同一 `BigFloat` object，或产生 false rejection；
   - Lower/Upper authority、allocating/in-place/try API 转发是否一致；
   - 同 workspace 不同 worker 的 task 并发是否安全，文档是否明确禁止同 worker 并发；
   - workspace 是否保留 matrix 引用或 MPFR pointer；
   - cold/warm benchmark 是否真实测量首次扩容与后续复用。

2. **Common factor protocol and solve validation**
   - Cholesky/LDLT/RRQR/LU 的 `factor_kind`, `factor_triangle`,
     `factor_failure_position`, `factor_diagnostics` 是否足以阻止 consumer 读私有字段；
   - vector/multi-RHS、precision、finite、alias、factor status 和 recorded-backend dispatch 是否在写入前正确验证；
   - test-only backend 是否明确 `UnsupportedOperation`，无 fallback。

3. **RRQR rank semantics**
   - 实现是否确实为 column-pivoted RRQR，满足 `A*P=Q*R`；
   - `threshold=max(atol,rtol*reference_scale)` 的 scale 定义、默认值与 extreme global scaling 是否一致；
   - `numerical_rank` 和 defensive metadata 是否在 mixed-precision/nonfinite corruption 下 fail closed；
   - `refine_once!` 是否在修改 residual/correction 前验证 tau/rank metadata。

4. **Diagnostics math**
   - Cholesky diagonal ratio、LDLT 1x1/2x2 normalized quality、RRQR accepted/rejected diagonal、LU swaps/permutation 是否数学定义清楚并按 factor precision 计算；
   - diagnostics 不得暗含 solver accept/reject policy。

5. **Native numerical trajectory and performance**
   - GEMM row/column view 与 Cholesky row-segment view 是否保持原 reduction order、MPFR precision 和 bitwise Legacy parity；
   - benchmark setup 是否在 timed region 外重建 mutable operands；
   - effective threads/block labels是否与真实 dispatcher 一致；
   - 不要把 n=8 wrapper timing 噪声当 correctness finding，但可以指出 benchmark methodology 的真实缺陷。

6. **Independence/provenance/release blockers**
   - `Project.toml`/`src` 是否无 SDPX dependency 和 solver concepts；
   - `THIRD_PARTY_NOTICES.md` 的 commit/blob/license 是否充分；
   - Julia/MPFR internal wrapper assumptions、CI 1.10/1.11/1.12 coverage、未测平台是否构成 release blocker。

## Verification already run

```text
Julia 1.12.6, --compiled-modules=no --check-bounds=yes --depwarn=yes
threads=1: 6258/6258 pass
threads=4: 6258/6258 pass
git diff --check: pass

SDPX Legacy bitwise parity, p=128/256/512:
dot, GEMM, Cholesky lower factor, Cholesky solve: pass

production benchmark: 108/108 gates, 108 measurements
standalone benchmark: 18/18 gates, 126 measurements
Legacy timing: 12/12 gates, 60 ratio cells
block/thread calibration: 48/48 gates, 186 measurements
```

详细数据见 `benchmark/results/2026-08-13-round-c.md`。本地没有对当前最终 ZIP 运行 Julia 1.10/1.11，也没有在目标 HPC 或 n>128 上完成同等全扫；请作为 residual risk，不要误写为已验证。

## Required output

先按 P0/P1/P2/P3 列 actionable findings。每项必须给：

- 文件和精确行号；
- 可复现输入/调用路径；
- 违反的具体 invariant；
- 最小修复建议；
- 必须新增的 regression test。

不要把纯风格偏好、无证据的算法重写或 roadmap 功能列为 finding。不要建议隐藏 fallback、自动 precision escalation、自动 refinement loop 或 solver policy。

最后分别回答：

1. 是否存在 correctness/ownership/precision blocker；
2. 是否存在隐藏 fallback；
3. factor protocol 是否足够稳定供 SDPX 薄适配器消费；
4. Native/Legacy 性能证据是否支持当前结论；
5. 是否建议标记 `API/CORRECTNESS FROZEN`；
6. 仍需在 Julia 1.10/1.11、HPC、large sizes 验证什么。

如果没有明显可改进项，请明确输出：

```text
NO ACTIONABLE FINDINGS
```

并仍列 residual risks。
