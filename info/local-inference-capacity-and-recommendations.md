# Local inference capacity and recommendations

ICN is the only authority for inference hardware, artifact inspection, model fit, model storage,
downloads, and active runtime state. CLI and web actions call ACN RPCs; ACN translates those actions
to the generated ICN client. ACN never treats its own host as the inference machine.

The curated model recipes are Magnitude-owned metadata. They group quantized choices under
stable checkpoint identities and record repositories, artifact selectors, product context
profiles, reviewed performance and fidelity evidence, and license policy. It does not pin Hugging
Face commits or copy filenames, shard lists, sizes, or hashes into source.

ICN queries Hugging Face, resolves `main` to an immutable snapshot, and returns current files,
sizes, identities, license data, and commit provenance. Preview and download then use that exact
commit. ICN derives GGUF architecture, parameter counts, quantization, maximum context, placement,
memory, and generation speed from artifact metadata. Arbitrary GGUF repositories use the same
resolution and preview path but have no curated Magnitude quality or fidelity claims.

Live discovery is cached by ICN: search results are brief, repository snapshots have a short TTL,
and GGUF headers plus fit/performance assessments are content-addressed by immutable artifact and
hardware evidence. No model weights are downloaded until the user chooses a model.

For each usage choice the ICN recipe service submits the applicable context and parallel-sequence
profiles to preview. Recommendations use only exact 100K and 200K fits with generation evidence at
that same context, and require at least 10 expected baseline tokens per second. The service ranks
eligible candidates into material Balanced, Best Quality, Fastest, and Lightweight intents using
common Terminal-Bench v2.1 capability, estimated generation speed, runtime memory, quantization
fidelity, and download size. Multiple quantizations of one checkpoint may appear when they explain
a real quality trade-off; duplicate filler cards are omitted. The UI always presents these intents as
Balanced, Best Quality, Fastest, then Lightweight, and explains each specialized option by comparing
its capability, speed, context, footprint, and possible quality loss with Balanced. The UI continues
to show recommendations, exact artifact details, hardware, download progress, downloaded models,
activation, restart, unload, and deletion. Download and load progress update the ICN inventory
snapshot, which ACN exposes through the ordinary mirrored-state contract.

Downloaded artifacts live in ICN's configured model store. `GET /v1/models` is the inventory and
residency authority, while `GET /v1/hardware` is the hardware and live memory authority. ACN persists
only user usage/profile and ordinary slot selections; it does not
persist a competing artifact index, endpoint binding, runtime installation, or active-model record.

The local provider ID is `local`. Its model catalog is projected from ICN inventory, demand loading
uses ICN runtime control, and generation streams through ICN chat. There is no external local-server
route or alternate model transport.
