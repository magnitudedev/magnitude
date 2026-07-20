# Local inference capacity and recommendations

ICN is the only authority for inference hardware, artifact inspection, model fit, model storage,
downloads, and active runtime state. CLI and web actions call ACN RPCs; ACN translates those actions
to the generated ICN client. ACN never treats its own host as the inference machine.

The curated catalog contains product and source policy only: display/family identity, immutable
repository source, primary artifact selector, context profiles to assess, quality ordering,
fidelity guidance, and license policy. Artifact size, hashes, shard membership, architecture,
parameter counts, quantization, maximum context, placement, and memory requirements come from
`POST /v1/models/preview`.

For each usage choice ACN submits the applicable context and parallel-sequence profiles to preview,
keeps only complete `Fits` results, and ranks them with the curated product policy. The UI continues
to show recommendations, exact artifact details, hardware, download progress, downloaded models,
activation, restart, unload, and deletion. Download and load progress are streamed from ICN and
projected into the existing ACN state.

Downloaded artifacts live in ICN's configured model store. `GET /v1/models` is the inventory
authority, `GET /v1/hardware` is the hardware authority, and `GET /v1/runtime/model` is the active
runtime authority. ACN persists only user usage/profile and ordinary slot selections; it does not
persist a competing artifact index, endpoint binding, runtime installation, or active-model record.

The local provider ID is `local`. Its model catalog is projected from ICN inventory, demand loading
uses ICN runtime control, and generation streams through ICN chat. There is no external local-server
route or alternate model transport.
