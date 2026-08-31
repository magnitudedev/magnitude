# Local inference capacity and ranking

ICN is the authority for inference hardware, artifact inspection, model fit, model storage,
downloads, and active runtime state. ACN projects those facts into Magnitude's local-model product
view; clients do not call ICN or maintain a competing inventory.

Magnitude's curated catalog defines reviewed model configurations and quality metadata. ICN
resolves immutable artifacts and derives architecture, parameterization, maximum context, placement,
memory, and generation-speed estimates. No model weights are downloaded until the user chooses a
model.

For each active catalog configuration, ICN assesses exact runtime memory and expected performance at
its reviewed context. ACN's local-model projection converts completed fitting assessments into
normalized intelligence, speed, and quality scores using a pure ranking policy. Speed compares
expected generation at 50K occupied context, bounded by the configured context for shorter models.
Scores belong to that exact configuration; incompatible, non-fitting, unresolved, deprecated, and
uncurated models do not receive them.

The setup UI combines those stable scores with two connection-scoped controls:

- Fast to Smart shifts preference between generation speed and intelligence while quality always
  contributes.
- Memory is a hard runtime-memory ceiling equal to the machine's normalized physical-memory domains.

The client filters by memory, ranks by the selected preference, and shows at most ten downloadable
models. Installed models remain separately available. The controls are not persisted and do not
change the full Settings catalog. Native load planning still rechecks current memory availability,
so a fitting ranked model can require closing other memory-intensive applications before loading.

Downloaded artifacts live in ICN's configured model store. The local provider ID is `local`; model
selection uses the canonical model ID, and generation streams through ICN chat. ACN persists slot
selection but not a competing artifact index, runtime installation, or active-model record.
