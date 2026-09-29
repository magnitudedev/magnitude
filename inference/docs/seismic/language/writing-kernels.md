# Writing kernels

Write the computation's useful structure directly: independent work, contribution domains, recurrences, access geometry and publication. Leave physical assignment and storage decisions to the compiler.

## Independent elementwise work

```seismic
fn add[M, N](x: &tensor[M, N] T, y: &tensor[M, N] U,
             result: &mut tensor[M, N] V):
    parallel for rows in 0..M:
        parallel for cols in 0..N:
            result[rows, cols] = f32(x[rows, cols]) + f32(y[rows, cols])
```

The two axes expose independent work even when one extent is small. Conversion and publication define numerical boundaries. The compiler can flatten, tile or distribute the domain while preserving those boundaries and exclusive writes. No workgroup-size hint belongs in this source.

## An ordered reduction and an authored alternative

```seismic
fn sum_any_order[N](t: tensor[N] f32) -> f32:
    return reduce(t, 0, sum)

fn sum_any_order[N](t: tensor[N] f32) -> f32:
    return reduce(t, 0, sum, unordered=true)
```

The first applicable body defines the reference reduction. The second exposes reassociation as a candidate choice. Numerical construction must establish where selecting it satisfies the entry's precision policy over all legal contents. Passing a few comparisons does not make it applicable.

The compiler must also support the reduction when its operand is a computed expression or view. Authors should not add artificial storage just to make one consumer's lowering succeed.

## Preserve the actual matrix contribution structure

```seismic
fn matmul[M, N, K](a: &tensor[M, K] T, b: &tensor[N, K] U,
                   into: tensor[M, N] f32) -> tensor[M, N] f32:
    let mut result = into
    for i in 0..M:
        for j in 0..N:
            let mut s = result[i, j]
            for k in 0..K:
                s = fma(f32(a[i, k]), f32(b[j, k]), s)
            result[i, j] = s
    return result
```

This body declares an ascending F32 FMA chain seeded by `into`. Physical windowing must carry the accumulator correctly. A matching native matrix facility must implement the authored contribution structure and satisfy the numerical contract. Replacing it with another factorization requires an authored alternative.

## Irregular writes and persistent model state

```seismic
fn kv_append[T, KV, W](key: &tensor[1, KV, W] A,
                       value: &tensor[1, KV, W] A,
                       history_key: &mut tensor[T, KV, W] A,
                       history_value: &mut tensor[T, KV, W] A,
                       destination: index[T]):
    parallel for heads in 0..KV:
        history_key[destination, heads] = key[0, heads]
        history_value[destination, heads] = value[0, heads]
```

The actual destination selects the row; its bound establishes legal indexing. The writes preserve the histories' allocation identities. The engine owns when sequence state advances; the kernel owns the declared updates. Persistent state here means state retained between operations or invocations, not disk storage.

## Authoring rules

- Preserve real intermediate rounding and required effect order, including across helper calls.
- Use parallel iteration for independent work and carries for recurrences. Do not encode independence through assumptions the checker cannot enforce.
- Supply alternate algorithms explicitly. Use typed capabilities when the algorithm needs a target-specific semantic operation.
- Express real input constraints. Do not add shape caps, special helper names or dummy copies to compensate for incomplete physical construction.
- Treat missing support for a legal composition as a compiler gap. The [general construction contract](../compiler/construction.md) must cover the language, including empty domains and mutation.
