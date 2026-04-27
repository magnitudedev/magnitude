# @magnitudedev/providers

Multi-provider LLM support with model slot abstraction for Magnitude.

## Canonical Models

Open source models with known HuggingFace repos are tracked as **canonical models** — provider-independent identities with chat templates. The canonical model registry lives in generated code and is driven by a hand-maintained manifest.

### Architecture

- **`model-manifest.ts`** — Hand-maintained source of truth. Lists each open model's `id`, `name`, `family`, and `hfRepo`.
- **`model/generated/`** — Auto-generated. Contains the registry (`canonical-model-registry.ts`), re-exports via `index.ts`, and template files in `templates/`.
- **`model/generated/templates/`** — Auto-generated. One directory per model containing `chat_template.jinja` fetched from HuggingFace.

### Fetching templates

After adding a new model to the manifest, run the fetch script to download its chat template and regenerate the registry:

```bash
cd packages/providers
bun run scripts/fetch-templates.ts
```

The script:
1. Reads the manifest
2. Fetches each model's chat template from HuggingFace (tries `tokenizer_config.json` first, then `chat_template.jinja`)
3. Writes template files to `src/model/generated/templates/{modelId}/chat_template.jinja`
4. Generates `src/model/generated/canonical-model-registry.ts`

### Adding a new model

1. Add an entry to `src/model/model-manifest.ts`:
   ```typescript
   { id: "new-model", name: "New Model", family: "family", hfRepo: "org/New-Model" },
   ```
2. Add the model to the relevant provider(s) in `src/registry.ts` using `providerModels()`
3. Run `bun run scripts/fetch-templates.ts`
4. `modelId` is auto-resolved via `tryResolveCanonicalModelId()` — if the model's `id` matches a manifest entry, it links automatically

### Importing templates

Template files use Bun's `with { type: "text" }` import syntax and are declared as string modules via `src/templates/templates.d.ts`. Vitest requires `assetsInclude` in `vitest.config.ts` to handle `.jinja` files.

## Testing

```bash
cd packages/providers && bunx --bun vitest run    # single run
cd packages/providers && bunx --bun vitest        # watch mode
```
