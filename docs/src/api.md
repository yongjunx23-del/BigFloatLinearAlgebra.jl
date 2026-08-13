# API

Every BLAS-style operation takes an explicit backend as its first argument.
`NativeBackend` is the default, `GenericBackend` is the reference.

## Backends and capabilities

```julia
capabilities(NativeBackend())
capabilities(GenericBackend())
```

`cholesky_triangles` in the returned capability tuple enumerates the
authoritative Cholesky triangles: `(:lower,)` for `NativeBackend`, and
`(:lower, :upper)` for `GenericBackend`.

## Storage

```julia
owned_zeros(BigFloat, m, n; precision_bits = 256)
owned_similar(A)
owned_copy(A)
copy_owned!(dst, src)
zero_owned!(A)
fill_owned!(A, value)
```

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
```

## Level 3

```julia
gemm!(backend, transA, transB, a, A, B, b, C)
syrk!(backend, triangle, trans, a, A, b, C)
trmm!(backend, side, triangle, trans, diagonal, a, A, B)
trsm!(backend, side, triangle, trans, diagonal, a, A, B)
```

## Cholesky

```julia
try_cholesky!(backend, A; triangle = Lower)
cholesky!(backend, A; triangle = Lower, check = true)
cholesky(backend, A; triangle = Lower, check = true)
ldiv!(factor, rhs)
solve!(factor, rhs)
solve(factor, rhs)
```
