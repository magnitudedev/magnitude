import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import { OPENAI_BASE_URL, defineConnector, launchPlan } from "../shared"
import { writeFileAtomic } from "../../utils/atomic-file"

const CODEX_BASE_INSTRUCTIONS = "You are a coding agent running in Codex CLI. Work with the user in the current workspace until their request is resolved. Inspect relevant files before changing them, follow repository instructions, make focused edits, verify consequential changes, and communicate progress and results concisely. Use the available tools when they are needed and preserve user work unrelated to the request."

const reasoningDescription = (effort: string): string => `Use the model's ${effort} reasoning effort.`

export const codexModelCatalog = (spec: HarnessConnectionSpec): string => `${JSON.stringify({
  models: spec.models.map((model, priority) => ({
    slug: model.id,
    display_name: model.name,
    description: model.description,
    ...(model.capabilities.reasoning.supported
      ? { default_reasoning_level: model.capabilities.reasoning.defaultEffort }
      : {}),
    supported_reasoning_levels: model.capabilities.reasoning.efforts.map((effort) => ({
      effort,
      description: reasoningDescription(effort),
    })),
    shell_type: "default",
    visibility: "list",
    supported_in_api: true,
    priority,
    additional_speed_tiers: [],
    service_tiers: [],
    availability_nux: null,
    upgrade: null,
    base_instructions: CODEX_BASE_INSTRUCTIONS,
    model_messages: null,
    supports_reasoning_summaries: false,
    default_reasoning_summary: "auto",
    support_verbosity: false,
    default_verbosity: null,
    apply_patch_tool_type: null,
    web_search_tool_type: "text",
    truncation_policy: { mode: "bytes", limit: 10_000 },
    supports_parallel_tool_calls: false,
    supports_image_detail_original: false,
    context_window: model.contextWindow,
    max_context_window: model.contextWindow,
    auto_compact_token_limit: null,
    comp_hash: null,
    effective_context_window_percent: 95,
    experimental_supported_tools: [],
    input_modalities: model.capabilities.vision ? ["text", "image"] : ["text"],
    supports_search_tool: false,
    use_responses_lite: false,
  })),
}, null, 2)}\n`

export const codexConfig = (spec: HarnessConnectionSpec, modelCatalogPath: string): string => {
  return `model_provider = "magnitude"
model_catalog_json = ${JSON.stringify(modelCatalogPath)}
service_tier = "default"
${Option.match(spec.setCurrent, {
  onNone: () => "",
  onSome: (modelId) => `model = ${JSON.stringify(modelId)}\n`,
})}
[model_providers.magnitude]
name = "Magnitude"
base_url = ${JSON.stringify(OPENAI_BASE_URL)}
wire_api = "responses"
requires_openai_auth = false
`
}

export const makeCodexConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "codex",
  name: "Codex",
  executable: "codex",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.codex, paths.codexModels],
  connect: (spec) => Effect.gen(function* () {
    yield* writeFileAtomic(paths.codexModels, codexModelCatalog(spec))
    yield* writeFileAtomic(paths.codex, codexConfig(spec, paths.codexModels))
  }),
  disconnect: () => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* Effect.forEach([paths.codex, paths.codexModels], (path) => fs.remove(path).pipe(
      Effect.catchTag("SystemError", (error) => error.reason === "NotFound" ? Effect.void : Effect.fail(error)),
    ), { discard: true })
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--profile", "magnitude"])
  },
})
