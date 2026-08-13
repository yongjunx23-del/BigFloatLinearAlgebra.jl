# BigFloatLinearAlgebra.jl final-review prompt

请解压并审查附件中的 `BigFloatLinearAlgebra.jl` 源码快照。这是一次独立的最终代码审查，不要仅复述 README，也不要默认已有修复正确。请以实际源码和测试为准。

## 项目定位

BFLA 是独立的 Julia `BigFloat` / MPFR 稠密线性代数包。必须保持：

- 显式 `NativeBackend` / `GenericBackend`；
- 无 Native 到 Generic 的隐藏 fallback；
- 无因子分解或求解的隐藏 fallback；
- 无自动精度变化；
- 普通运算严格同精度，跨精度只能走显式转换 API；
- 每个可变 MPFR destination 槽位独立所有；
- 权威三角语义；
- factor operation 必须通过 factor 记录的 backend；
- 不引入 SDPX、KKT、Schur、cone 或 solver policy。

## 本轮修复范围

本快照从已审查基线 `e9971f9` 继续，包含以下实现提交：

```text
12ac7ea Fix symmetric kernel dimension validation
8a80405 Enforce factor ownership and backend identity
fe6fc23 Make ownership mutations precision strict
93d583f Validate config and narrow workspace contract
```

此外，快照包含与实现一致的 `CHANGELOG.md` 和 `docs/src/*.md` 契约更新。

重点重新检查：

1. `gemmt!` / `syr2k!`
   - 所有 transpose 组合的完整 outer 和 contraction dimension 是否正确；
   - malformed dimensions 是否在任何 `@inbounds` 和 destination 写入前失败；
   - Native/Generic、Lower/Upper 是否都被测试；
   - 失败后 `C` 是否保持数值和对象 identity 不变。

2. LDLT MPFR ownership
   - Native/Generic 的 1x1、2x2、trailing update、permutation 路径是否仍存在 `BigFloat` 槽位浅赋值；
   - `_swap_sym!` 是否只是双射置换，不会复制对象引用；
   - `ldlt!` 对预先共享的 authoritative lower 槽位是否在写入前 fail closed；
   - non-authoritative upper 中的 stale/NaN/shared 对象是否仍被正确忽略并重建；
   - `_require_independent_triangle_elements` 的 `objectid`、排序和 collision 复核是否严格且跨 Julia 1.10/1.12 可用；
   - allocating `ldlt` 是否通过 ownership-safe deep copy 接受预别名 source。

3. Factor backend identity
   - Cholesky、LDLT、QR、LU solve，以及 QR `applyQ!`，是否都经 factor 记录的 backend 分派；
   - test-only backend 是否在任何 RHS/target 写入前抛 `UnsupportedOperation`；
   - 是否存在 backend-independent helper 绕过分派或任何隐式 fallback。

4. Precision/failure atomicity
   - `fill_owned!` 是否完整预检 destination 与 value 精度后才写入；
   - `mirror_triangle!` 是否完整预检矩阵精度后才写入；
   - mixed precision 失败是否保持数值和对象 identity 不变；
   - 是否仍有普通 API 暗中承担 precision conversion。

5. Config/workspace contract
   - `KernelConfig` 是否拒绝 `thread_count < 1` 和负 block size；
   - `0` 是否明确且真实表示 unblocked；
   - 未实现的 `ldlt_block` 是否彻底移除；
   - `BFLAWorkspace` 是否准确收窄为 caller-managed scratch；
   - 公开 kernel 是否仍接受但忽略 `workspace=`；
   - 删除旧 keyword/field 的 API 兼容影响是否需要版本或迁移说明。

6. 测试质量
   - fixture 本身是否避免 chained `BigFloat` assignment 和浅复制；
   - ownership probe 是否真的能检测对象共享；
   - tests 是否有只验证实现细节、零值假阳性、漏掉矩阵 RHS 或失败前写入的问题；
   - 是否缺少足以阻止发布的回归用例。

## 已执行验证

当前快照在 Julia 1.12.6 上执行：

```text
Pkg.test(), threads=1, --check-bounds=yes, --depwarn=yes: 5707/5707 pass
Pkg.test(), threads=4, --check-bounds=yes, --depwarn=yes: 5707/5707 pass
git diff --check: pass
```

Julia 1.10.11 在本轮最后一组纯测试增强之前已通过同一实现的 1/4 线程测试；请不要把这视为当前 ZIP 的完整 1.10 复验。

LDLT 256-bit 强制 2x2 fixture 的 allocation 中位数（10 samples）：

```text
n=4:   baseline 4560,    current 4704 bytes
n=32:  baseline 221856,  current 227040 bytes
n=128: baseline 3782752, current 3864736 bytes
```

本轮未执行完整 runtime benchmark，也未运行 Julia 1.11。

## 输出要求

请先列 findings，按 P0/P1/P2/P3 严重度排序。每项必须包含：

- 严重度和简短标题；
- 具体文件与行号；
- 可复现反例或清晰的错误路径；
- 为什么违反 BFLA 契约；
- 最小修复建议；
- 应新增或修改的回归测试。

不要把纯风格偏好列为 finding。不要建议大规模重写、自动 fallback、自动 precision escalation 或新 roadmap 功能。

随后分别给出：

1. 对上述六个审查领域的结论；
2. 未验证项和残余风险；
3. 是否建议进入下一次发布审查。

如果没有明显可改进项，请明确写：

```text
NO ACTIONABLE FINDINGS
```

并仍然列出测试/平台覆盖方面的残余风险。
