# Direct native implementations

`elementwise.seismic` keeps an ordinary Seismic definition for each
operation and attaches one top-level Metal implementation. Generated callers
choose the path explicitly:

```rust
let portable = kernels::add_f32::for_device(&device, precision)?;
let native = kernels::add_f32::native_for_device(&device)?;

// The same generated Args type and call shape are used by both.
native.call(kernels::add_f32::Args {
    x: &x,
    y: &y,
    result: &mut result,
})?;
```

The Metal source receives generated `SEISMIC_*` ABI macros. Tensor buffer
macros use the source parameter name (`SEISMIC_BUFFER_X`) and mutable tensor
parameters are passed in place. Runtime dimensions and tensor metadata are
available through `seismic_words`; dimensions have named accessors such as
`SEISMIC_DIM_M`. Owned tensor results use `SEISMIC_RESULT_<ordinal>_BUFFER`,
and scalar results use `SEISMIC_RESULT_<ordinal>_WORD` in
`SEISMIC_BUFFER_SCALAR_RESULTS`.

The `NativeKernel` handle supports direct `.call(...)` only. It cannot be
enqueued into a Seismic workflow and never enters lowering, tuning, solving,
schedule construction, or portfolio selection.

`runner` is an executable end-to-end example. On macOS, run it with
`cargo run --manifest-path seismic/examples/native/runner/Cargo.toml`.
