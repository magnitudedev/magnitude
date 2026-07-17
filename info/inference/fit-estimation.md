# llama.cpp fit estimation

Magnitude uses `llama-fit-params --fit-print on` as a pre-load memory estimate for a concrete
model artifact and execution profile. The output is useful, but it is not an exact measurement and
it is not the same operation as loading the model.

## What upstream reports

`llama-fit-params` opens the primary GGUF without allocating its tensors, constructs llama.cpp's
model and context graphs, and prints estimated model, context, and compute memory for each backend.
Magnitude invokes the `llama-fit-params` executable from the same validated llama.cpp installation
as `llama-server` and passes only arguments accepted by that executable.

The standalone tool does not accept `--mmproj`. It therefore does not include multimodal projector
weights or projector compute buffers. It also does not accept every server option; notably, the
tested b10011 executable rejects `--kv-unified`. A successful result describes the closest profile
the standalone tool can represent, not necessarily every detail of the eventual server process.

`--fit-print` also bypasses the fitting search performed by a normal `llama-fit-params` invocation.
It reports the requested/default placement. The server remains responsible for applying `--fit`
during a real load.

## Magnitude estimate

For a text-only artifact:

```text
estimated bytes = sum of every reported model, context, and compute placement
```

For an artifact with an associated projector:

```text
base bytes              = sum of every reported placement
projector estimate      = ceil(projector file bytes * 1.20)
estimator uncertainty   = 1.5 GiB
estimated bytes         = base bytes + projector estimate + estimator uncertainty
```

The projector is identified by the artifact's declared file role. Filenames and model-family names
are not used. The constants are identified by one estimation-policy fingerprint and must be changed together with
tests and this document when broader measurements justify a revision.

The 1.20 projector factor covers projector weights plus observed working memory. The fixed 1.5 GiB
reserve covers the largest observed difference between standalone fit output and loaded
language-model buffers without multiplying an already-large model estimate.

## Capacity comparison

Fit assessment uses stable hardware capacity, not point-in-time free memory:

- reserve the larger of 8 GiB or 20% from system memory;
- reserve the larger of 1 GiB or 10% from each discrete accelerator;
- on Apple Silicon, treat Metal and host allocations as one unified physical-memory domain and
  compare the complete estimate once against stable system capacity;
- on discrete-memory systems, compare each reported placement with its device, and conservatively
  charge the vision adjustment to the primary accelerator (or host when there is no accelerator).

Current free memory is deliberately not catalog membership or model availability. Memory pressure
can change between discovery and acquisition. The actual `llama-server --fit` and load attempt is
authoritative and may still fail even after a positive estimate.

## Empirical basis

Measurements on an Apple M4 Max with llama.cpp b10011 covered Gemma 4 E2B, E4B, and 26B-A4B plus
Qwen 3.6 27B, at 32K and 100K contexts and one or three parallel slots.

| Model and profile | Combined observed/estimated requirement | Magnitude estimate | Difference |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B Q4, 32K x 1 | 6,358 MiB | 6,775 MiB | +6.6% |
| Gemma 4 E4B Q4, 32K x 1 | 8,855 MiB | 8,964 MiB | +1.2% |
| Gemma 4 26B-A4B Q4, 32K x 1 | 19,240 MiB | 21,317 MiB | +10.8% |
| Qwen 3.6 27B Q4, 32K x 1 | 20,296 MiB | 21,467 MiB | +5.8% |
| Gemma 4 E2B Q4, 100K x 1 | 6,884 MiB | 7,300 MiB | +6.0% |
| Gemma 4 E4B Q4, 100K x 3 | 10,082 MiB | 10,105 MiB | +0.2% |

The combined reference is loaded language-model buffer accounting plus llama-server's internal
worst-case projector reservation. Matched text-only/multimodal RSS deltas independently checked the
projector component.

Observed projector RSS increases were 1.10-1.19 times projector file size. llama-server's internal
projector estimate was between 0.3% low and 11.8% high. Raw standalone fit output underestimated
the complete multimodal requirement by 4-35%, largely because its main-model estimate varied by
architecture. The Magnitude additive formula was approximately 0-11% above the combined observed
requirement across the six tested configurations.

These measurements cover two current multimodal families on Metal. They do not prove a universal
error bound for every projector architecture, backend, multi-GPU topology, speculative model, or
future llama.cpp build.

## Appropriate use

The estimate is appropriate for ordering, warnings, and explaining why a configuration may be
unlikely to fit. It is not an allocation measurement, a live-pressure oracle, an availability
gate, or proof that loading will succeed. A capacity-risk result remains selectable and loadable.
Loaded and sleeping routes are authoritative evidence and are not invalidated by a later pre-load
estimate.

## Scheduling and caching

Fit analysis is background derivation, not catalog construction. After artifact inspection, one
backend coordinator reconciles required assessment keys against a persisted cache. An assessment
key includes the normalized artifact path and complete file version, execution profile, fit-tool
fingerprint, stable hardware/backend topology, and estimation-policy fingerprint.

Missing or stale assessments run in one batch with at most two `llama-fit-params` subprocesses at a
time. Catalog and UI state publish immediately with no assessment; neither waits for this batch.
Each completed current result is persisted and causes a lightweight state publication. Equivalent
work is single-flight, and stale results are discarded if their key changed while running.

Reconciliation occurs after startup artifact inspection and after revisions to artifacts, serving
profiles, the selected llama.cpp installation, stable hardware topology, or the estimation policy.
There is no timer, frontend trigger, selection trigger, or invocation during catalog reads. An
explicit model load never waits for fit analysis.
