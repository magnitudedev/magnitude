# icn-fit

`icn-fit` runs the exact `common_get_device_memory_data` and `common_fit_params` implementation from
the llama.cpp revision nested in our bindings fork. It reads GGUF metadata and builds no-allocation
graphs, but does not load model tensor data. This makes it suitable for admission checks before an
ICN loads a model.

```sh
cargo run --manifest-path inference/Cargo.toml -p icn-fit -- \
  --model /absolute/path/to/model.gguf
```

The JSON report contains the requested and fitted context/GPU layers, tensor splits and overrides,
per-device model/context/compute estimates, margins and deficits, plus typed adjustments and
warnings. Flags for context, batching, sequence count, KV types, Flash Attention, KV offload, SWA,
and unified KV must match the context ICN intends to create.

## Boundaries of the pinned estimator

The report is a planner estimate, not observed peak RSS or runtime allocation. The pinned upstream
implementation does not account for a multimodal projector, draft/MTP model, other models already
loaded by ICN, transient model-load allocations, or an OS reserve beyond the supplied margins. It
also assumes host memory is unlimited whenever an accelerator is present. Device free memory is a
point-in-time reading, and integrated/unified-memory backends may report the same physical pool in
more than one role.

`common_fit_params` returns only success/failure/error; its detailed fit failure text is emitted to
the llama logger and is not a structured API. We intentionally do not scrape that text. Exceptions
from the initial and fitted structured measurement calls are retained as diagnostics.
