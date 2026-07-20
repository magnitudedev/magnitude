# Providers

Provider system spans three packages:
- `packages/ai` — provider-agnostic contract (`Provider`, `ModelCatalog`, `BoundModel`, `BaseCallOptions`).
- `packages/providers` — concrete implementations + registry/aggregation.
- `packages/sdk` — `ProviderClient`: sole consumer boundary. Consumers never import from `ai` or `providers` directly.

## Provider Contract

`Provider<TModel>`: `id`, `displayName`, `catalog`, `discoverModelProperties(request)`, `bindModel(id, options?)`, `classifyModelFamily(model)`. Hosted providers return their authoritative resolved properties immediately; runtime providers may perform demand-driven discovery. Extensions are separate interfaces: `WebSearchExtension`, `UsageExtension`.

`BaseCallOptions`: `maxTokens`, `toolChoice`, `reasoningEffort`, `generateToolCallId`. Provider-specific options are baked at `bindModel` time via `wrapAsBaseModel` — consumers only see `BoundModel<BaseCallOptions>`.

Catalog eligibility is provider-specific. Hosted catalogs may exclude models
they do not support. The local provider consumes an ACN-owned catalog and
scoped acquisition source; it does not discover endpoints, files, or processes.
Unavailable local models remain visible with a typed disabled reason.

## Model Family Classification

All providers share one `classifyModelFamily` function from the family registry. Do not write provider-specific classifiers.

The shared classifier tokenizes model IDs into atoms and matches them against pattern definitions using resilient symbols (`lit`, `sep`, `dot`, `num`, `ver`, `opt`). Patterns match at any position in the atom stream, so the same pattern matches model IDs with different prefixes/suffixes across providers.

Rules:
- Use flexible symbols, not exact string matching. `num`/`ver` match any numeric/version segment. `opt` matches optional segments. This lets one pattern cover variants like `glm-5.1`, `glm-5.2`, `provider/glm-5.1-instruct`.
- Add new families by extending `FAMILY_DEFINITIONS` in the family registry, not by adding per-provider logic.
- Family classification enriches catalog metadata; it is not an availability gate. Local aliases, served IDs, display names, and filenames are arbitrary and must not be treated as family evidence. Local classification uses parsed architecture/tokenizer/base-model metadata only and otherwise omits `modelFamilyId`; do not use an `"unknown"` sentinel.
- The classifier must not add provider-specific matching. Patterns use resilient symbols that are consistent and flexible enough to match the same family across different provider ID formats.

## New Provider Steps

Hosted providers generally use files under `packages/providers/src/<name>/`: `contract.ts`, `catalog.ts`, `models.ts`, `errors.ts`, `provider.ts`, `index.ts`. Local runtime providers may instead receive a product-owned `ModelCatalog` plus scoped acquisition closure.

1. `contract.ts` — Effect Schema-derived serializable model info, option types, and error shapes. The local model is transport-independent; ICN runtime facts stay behind its ACN-owned source.
2. `catalog.ts` — `ModelCatalog<TModel>`: fetch endpoint, classify with `classifyModelFamily`, filter unclassified, TTL cache. Use `makeFileBackedModelCatalog` for cross-process cache.
3. `models.ts` — `NativeChatCompletions.model(...)` for OpenAI-compatible providers. `Option.define` for provider-specific options. `wrapAsBaseModel` to hide them.
4. `errors.ts` — `classifyRejectedResponse` maps provider errors to `ProviderRejection` variants.
5. `provider.ts` — construct a `Provider<TModel>` (plus status/extensions where applicable). Export `PROVIDER_ID`. A local provider's bound stream must hold its acquired runtime lease until the response event stream terminates.
6. Register in `registry.ts`, export from `index.ts`.
7. Wire into `createProviderClient()` in `packages/sdk/src/provider-client.ts`.

Non-OpenAI-compatible providers: custom protocol namespace + `Model.define` directly.
