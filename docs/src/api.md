# API

Every BLAS-style operation takes an explicit backend as its first argument.
`NativeBackend` is the default, `GenericBackend` is the reference.

## Backends and capabilities

```julia
capabilities(NativeBackend())
capabilities(GenericBackend())
```

`cholesky_triangles` in the returned capability tuple enumerates the
authoritative Cholesky triangles using the public `Triangle` enum: `(Lower,)`
for `NativeBackend`, and `(Lower, Upper)` for `GenericBackend`.
`cholesky_workspace = true` states that the backend accepts the explicit
ownership-scan workspace contract described below.
Both backends explicitly report column-pivoted rank-revealing QR with
`unpivoted_qr = false`, `rank_revealing_qr = true`, and
`qr_pivoting = :column`.

## Common factor protocol

All BFLA factors support:

```julia
factor_matrix(F)
factor_backend(F)
factor_precision(F)
factor_status(F)
factor_kind(F)
factor_triangle(F)
factor_failure_position(F)
factor_diagnostics(F)
issuccess(F)
```

`factor_triangle` returns `Lower` or `Upper` for Cholesky, `Lower` for LDLT,
and `nothing` for RRQR and LU. Consumers should not read concrete factor fields
such as `status.position`, `backend`, or factor-specific packed metadata.
`factor_diagnostics` returns numerical facts only. Cholesky reports its
authoritative triangle, failure position, minimum/maximum absolute factor
diagonal, and their ratio. LDLT reports inertia, pivot counts, minimum absolute
1x1 pivot, minimum absolute 2x2 determinant, and minimum normalized 2x2 quality
`abs(det(Dblock))/max(abs(d11),abs(d12),abs(d22))^2`. RRQR reports its rank
policy, defensive `R` diagonal/permutation copies, the minimum accepted
diagonal, and the next rejected diagonal when present. LU reports row swaps,
permutation, and failure position.

## Storage

```julia
owned_zeros(BigFloat, m, n; precision_bits = 256)
owned_similar(A)
owned_copy(A)
convert_owned!(dst, src)
copy_owned!(dst, src)
zero_owned!(A)
fill_owned!(A, value)
```

`fill_owned!` accepts only a `BigFloat` value whose precision matches every
destination element. Precision validation is completed before mutation.

## Configuration and caller scratch

```julia
KernelConfig(
    thread_count = 1,
    gemm_block = 0,
    syrk_block = 0,
    cholesky_block = 0,
    trsm_block = 0,
)
BFLAWorkspace(precision_bits; workers, scalar_slots)
workspace_scratch!(workspace, worker, slot)
workspace_buffer!(workspace, worker, length)
```

`thread_count` must be positive and block sizes must be non-negative. A zero
block size selects the unblocked Native kernel. `BFLAWorkspace` is explicitly
caller-managed storage. Cholesky alone can explicitly consume its worker-local
identity buffer to reuse ownership-scan allocation; no other kernel accepts or
silently ignores a workspace keyword.

## Level 1

```julia
scal!(backend, a, x)
axpy!(backend, a, x, y)
axpby!(backend, a, x, b, y)
dot(backend, x, y)
norminf(backend, x)
```

## Level 2

```julia
gemv!(backend, trans, a, A, x, b, y)
trsv!(backend, triangle, trans, diagonal, A, b)
syr!(backend, triangle, a, x, A)
symv!(backend, triangle, a, A, x, b, y)
```

## Level 3

```julia
gemm!(backend, transA, transB, a, A, B, b, C)
syrk!(backend, triangle, trans, a, A, b, C)
gemmt!(backend, triangle, transA, transB, a, A, B, b, C)
syr2k!(backend, triangle, trans, a, A, B, b, C)
trmm!(backend, side, triangle, trans, diagonal, a, A, B)
trsm!(backend, side, triangle, trans, diagonal, a, A, B)
```

## Cholesky

```julia
try_cholesky!(backend, A; triangle = Lower, workspace = nothing,
              workspace_worker = 1)
cholesky!(backend, A; triangle = Lower, check = true, workspace = nothing,
          workspace_worker = 1)
cholesky(backend, A; triangle = Lower, check = true, workspace = nothing,
         workspace_worker = 1)
ldiv!(factor, rhs)
solve!(factor, rhs)
solve(factor, rhs)
```

In-place Cholesky requires independent `BigFloat` objects in the authoritative
triangle and checks that ownership precondition before numerical mutation.
Sharing or poisoned values confined to the inactive triangle are ignored.
Allocating `cholesky` deep-copies the source and therefore repairs source-side
element sharing. When supplied, the workspace precision must equal matrix
precision. Concurrent calls sharing one workspace must reserve distinct worker
slots; a workspace never changes the backend or numerical algorithm.

## LDLT

```julia
try_ldlt!(backend, A)
ldlt!(backend, A; check = true)
ldlt(backend, A; check = true)
factor_perm(factor)
factor_blocks(factor)
factor_inertia(factor)
factor_diagnostics(factor)
```

## Rank-revealing QR

```julia
qr!(backend, A; atol = nothing, rtol = nothing)
qr(backend, A; atol = nothing, rtol = nothing)
applyQ!(factor, rhs, NoTrans)
applyQ!(factor, rhs, Trans)
factor_jpvt(factor)
factor_rank(factor)
numerical_rank(factor; atol = ..., rtol = ...)
factor_Rdiag(factor)
factor_tolerance(factor)
factor_rank_atol(factor)
factor_rank_rtol(factor)
factor_rank_scale(factor)
factor_rank_threshold(factor)
```

This is column-pivoted rank-revealing QR (`factor_kind(factor) == :rrqr`) with
`A*P = Q*R`; it is not unpivoted QR. Rank uses
`max(atol, rtol*reference_scale)`, where `reference_scale` is the largest
input-column 2-norm. Defaults are zero `atol` and
`max(size(A)...)*eps(BigFloat)` for `rtol`, both constructed at factor
precision. The legacy `tol` keyword remains as an absolute-only compatibility
mode and cannot be combined with `atol` or `rtol`. `numerical_rank` re-evaluates
rank from the packed `R` and recorded input scale without exposing mutable
factor storage.

For an overdetermined factor (`m >= n`), `solve!`
overwrites the first `n` rows of each RHS with the pivoted basic solution. The
remaining rows retain transformed residual information. Underdetermined
in-place solve is rejected because an `m`-row RHS cannot hold an `n`-entry
solution without changing shape.

LDLT solve, QR Q application, and QR solve always use the backend recorded in
the factor. There is no operation-level fallback to another backend.

## Partial-pivoting LU

```julia
try_lu!(backend, A)
lu!(backend, A; check = true)
lu(backend, A; check = true)
factor_pivots(factor)
factor_perm(factor)
factor_diagnostics(factor)
```

LU factors square dense matrices as `P*A = L*U`; `factor_pivots` reports the
stepwise row swaps and `factor_perm` reports the final row ordering. Both
accessors return defensive copies. LU is explicitly requested and is never
used as a fallback for another factor type.

## Residual and backward error

```julia
residual!(backend, trans, A, x, b, residual)
residual!(backend, A, x, b, residual)
normwise_backward_error(backend, trans, A, x, b, residual)
normwise_backward_error(backend, trans, A, x, b)
higher_precision_residual!(
    backend, trans, A, x, b, residual;
    residual_precision = q,
)
```

The residual API writes `b - op(A)*x` into ownership-safe caller storage and
supports vector or multi-RHS inputs. `normwise_backward_error` reports
`norm(r, Inf)/(norm(op(A), Inf)*norm(x, Inf)+norm(b, Inf))`, using induced
infinity norms for matrices. It reports a numerical fact and does not impose
an accept/reject threshold.

`higher_precision_residual!` is the explicit cross-precision path: `A`, `x`,
and `b` share factor precision `p`, while the required `residual_precision=q`
must match every element of caller-owned `residual` and satisfy `q > p`. It
promotes the input values into q-bit arithmetic and returns a
diagnostic tuple with `factor_precision`, `residual_precision`, and q-bit
`backward_error`. It never changes factorization, backend, precision, or
refinement policy.

```julia
refine_once!(factor, A, x, b, residual, correction)
```

`refine_once!` performs exactly one residual/correction/update step through the
backend recorded by `factor`. `A`, `x`, `b`, and caller-owned `correction` use
factor precision p; caller-owned `residual` may use p or q>p. Diagnostics report
the backward error before and after the step. Transpose refinement is rejected
until factors expose a transpose-solve contract, and BFLA never loops or chooses
an acceptance threshold. For RRQR, packed storage, Householder coefficients,
and all rank-policy metadata are checked for recorded precision and finiteness
before any of these caller-owned buffers can be modified.
