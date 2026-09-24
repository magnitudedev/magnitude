---
applies_to:
  - inference-v4/engine/model-executor/src/residency.rs
  - inference-v4/engine/model-executor/src/resident_weights.rs
---

# Device residency

The execution plan identifies every source tensor, artifact component, resident representation,
and byte charge before a device is opened. ResidencyStore is the sole importer and cache type for
device weights. Its key contains artifact identity, tensor name, and resident representation with
its layout, so tied weights share physical storage while distinct components, representations, and
layouts remain separate. The layout is the one fastest for the opened backend's kernels and may
differ by backend: one map picks the representation from the source format and the layout from
the execution path and backend (native Metal and Vulkan `rows16`, native CUDA `mma16`, native CPU and planned
`packet`). Import converts through the weight's `[B, N, K]` view, which keeps every layout's row
geometry; it never flattens a weight.

Each import takes an immutable artifact source, validates its exact WeightPlan, allocates the
planned resident destination and a one-shot source upload, submits the attested import program,
and publishes the resident weight only after completion. Failed imports leave no cache entry.

The target component is imported before engine readiness. Enabled optional head and vision
components are held by typed one-shot ComponentLoaders. Each loader owns its import store and
caches either its assembled component or its typed failure. The head loader inherits the target
store so tied embedding and output weights retain their exact resident tensors; the vision loader
owns an isolated projector store. Numerical stages receive loaders rather than shared mutable
cache access. No warm token path prepares or searches for an import kernel.
