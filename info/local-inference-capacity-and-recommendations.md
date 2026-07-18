# Local inference capacity and recommendations

Magnitude recommends local GGUF models from stable machine capacity, the
capabilities exposed by the selected llama.cpp installation, and
artifact-specific runtime metadata.

## Hardware capacity

The operating system supplies platform, native architecture, total system
memory, CPU identity, and logical core count. The selected `llama-server`
binary supplies the accelerator devices and memory it can actually use through
`--list-devices`.

The selected runtime is always the Magnitude-managed llama.cpp distribution.
ACN does not select binaries from `PATH`, accept a user-provided executable, or
connect local inference to a separately operated llama.cpp server.

Device memory is normalized into physical-device or unified-memory domains:

- Apple Silicon Metal memory aliases the system's unified memory instead of
  being added as separate VRAM.
- Discrete CUDA, ROCm, Vulkan, and SYCL devices retain separate capacity.
- A physical GPU exposed through more than one llama.cpp backend is counted
  once, with native compute backends preferred over Vulkan.
- Multiple devices are combined only when llama.cpp exposes them through one
  backend that supports the configured layer split.

Recommendations use total stable capacity, not point-in-time free memory.
Current free memory is observational only. Stable capacity reserves the larger
of 8 GiB or 20% of system memory and the larger of 1 GiB or 10% of discrete
accelerator memory.

## Catalog memory model

Every catalog artifact records its exact files and byte sizes plus runtime
metadata read from the pinned GGUF revision:

- GGUF architecture and exact parameter count
- transformer block and embedding dimensions
- attention head, key, and value dimensions
- full-attention and sliding-attention layer groups
- recurrent-state dimensions for hybrid attention/SSM architectures

The pre-download estimate includes the GGUF weights once, conservative graph
and compute overhead, F16 KV cache for every reserved context slot, and
per-slot recurrent state. Sliding-attention layers retain only their declared
window. This avoids scaling all model bytes with context length and avoids
treating modern hybrid architectures like conventional full-attention
transformers.

The recommendation policy evaluates 200K and 100K contexts. One-session usage
reserves one slot; up-to-three-session usage reserves three uniform slots.
Configurations that do not fit stable capacity are excluded. The primary
choice favors catalog model quality, then accelerator placement, context, quant
fidelity, and remaining capacity. Up to three cards are returned: the primary,
a smaller model, and a useful alternative or higher-fidelity quant.

## Download and activation

A managed llama.cpp distribution lives below
`~/.magnitude/local-inference/llamacpp/distribution`. New model downloads live
below `~/.magnitude/models` using Hugging Face's content-addressed
snapshot/blob layout. Magnitude's small publication manifests live below
`~/.magnitude/models/.manifests`.

The model registry also discovers existing GGUF files read-only from the legacy
Magnitude model directory and the user's Hugging Face cache. Those files can be
run by Magnitude's managed llama.cpp runtime, but Magnitude does not claim
ownership or delete them.

A recommendation ID identifies the exact artifact, context, parallel-slot
count, and runtime profile. The daemon re-resolves that ID against the current
hardware and usage selection before downloading. It persists the selected
profile, uses it when constructing the managed llama.cpp request, and records
the same context and slot count when the model is activated.

After files exist locally, `llama-fit-params` is the authoritative fit check.
It uses the resolved artifact, selected execution profile, selected llama.cpp
binary, and current hardware topology. Catalog estimates therefore guide the
download decision; llama.cpp validates the concrete launch.

## Performance estimates

Memory fit does not imply a trustworthy tokens-per-second prediction.
Throughput depends on memory bandwidth, backend kernels, offload placement,
prompt length, batch shape, and the difference between prompt processing and
token generation. Magnitude should expose performance estimates only after
collecting calibrated llama.cpp benchmark results for the relevant hardware,
artifact, quant, context occupancy, and slot count. It should report prompt
processing and generation throughput separately rather than deriving either
from parameter count.
