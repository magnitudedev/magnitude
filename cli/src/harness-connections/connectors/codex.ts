import * as Command from "@effect/platform/Command"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Option, Schema } from "effect"
import type { HarnessConnectionSpec } from "../contract"
import type { HarnessConnectionPaths } from "../paths"
import {
  CODEX_PROXY_BASE_URL,
  codexLocalModelId,
  defineConnector,
  isCodexLocalModelId,
  launchPlan,
  qualifiedModelSelection,
  readOr,
  removeTomlTable,
  splitQualifiedModelSelection,
  tomlTopLevelValue,
  updateTomlTopLevel,
  writeIfChanged,
} from "../shared"
import { removeConfigurationFile, writeFileAtomic } from "../configuration-file"

const CODEX_BASE_INSTRUCTIONS = "You are a coding agent running in Codex CLI. Work with the user in the current workspace until their request is resolved. Inspect relevant files before changing them, follow repository instructions, make focused edits, verify consequential changes, and communicate progress and results concisely. Use the available tools when they are needed and preserve user work unrelated to the request."
const CODEX_PROVIDER_ID = "magnitude"
const CODEX_PROVIDER_TABLE = `[model_providers.${CODEX_PROVIDER_ID}]
name = "OpenAI"
base_url = ${JSON.stringify(CODEX_PROXY_BASE_URL)}
wire_api = "responses"
requires_openai_auth = true
supports_websockets = true
supports_standalone_web_search = true
`

const CodexBundledCatalogDocumentSchema = Schema.Record({
  key: Schema.String,
  value: Schema.Unknown,
})
export type CodexBundledCatalog = Readonly<Record<string, unknown>> & {
  readonly models: ReadonlyArray<unknown>
}
export type CodexBundledCatalogReader = (
  executable: string,
) => Effect.Effect<CodexBundledCatalog, unknown, CommandExecutor.CommandExecutor>

class CodexBundledCatalogInvalid extends Data.TaggedError("CodexBundledCatalogInvalid")<{
  readonly message: string
}> {}

const reasoningDescription = (effort: string): string => `Use the model's ${effort} reasoning effort.`

const magnitudeCodexModels = (spec: HarnessConnectionSpec) => spec.models.map((model, priority) => ({
  slug: codexLocalModelId(model.id),
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
}))

export const codexModelCatalog = (
  spec: HarnessConnectionSpec,
  bundled: CodexBundledCatalog,
): string => `${JSON.stringify({
  ...bundled,
  models: [...bundled.models, ...magnitudeCodexModels(spec)],
}, null, 2)}\n`

const readBundledCatalog = (executable: string) => Command.make(
  executable,
  "debug",
  "models",
  "--bundled",
).pipe(
  Command.string,
  Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(CodexBundledCatalogDocumentSchema))),
  Effect.flatMap((document) => Array.isArray(document.models)
    ? Effect.succeed(document as CodexBundledCatalog)
    : Effect.fail(new CodexBundledCatalogInvalid({
        message: "Codex bundled catalog must contain a models array",
      }))),
)

const removeMagnitudeProvider = (source: string): string =>
  removeTomlTable(source, "model_providers.magnitude")

const installMagnitudeProvider = (source: string): string => {
  const withoutProvider = removeMagnitudeProvider(source).trimEnd()
  return `${withoutProvider}${withoutProvider.length === 0 ? "" : "\n\n"}${CODEX_PROVIDER_TABLE}`
}

export const makeCodexConnector = (
  paths: HarnessConnectionPaths,
  bundledCatalog: CodexBundledCatalogReader = readBundledCatalog,
) => defineConnector({
  id: "codex",
  name: "Codex",
  executable: "codex",
  requiresStartup: true,
  skillInstallationTarget: "shared-agents",
  configurationFiles: [paths.codex, paths.codexUser, paths.codexModels],
  connect: (spec) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const bundled = yield* bundledCatalog(spec.installation.executable)
    const userSource = yield* readOr(paths.codexUser, "")
    const previousModel = tomlTopLevelValue(userSource, "model")
    const previousProvider = tomlTopLevelValue(userSource, "model_provider")

    yield* writeFileAtomic(paths.codexModels, codexModelCatalog(spec, bundled))
    let next = removeMagnitudeProvider(userSource)
    next = updateTomlTopLevel(next, [
      ["model_provider", CODEX_PROVIDER_ID],
      ["model_catalog_json", paths.codexModels],
    ])
    if (Option.isSome(spec.model)) {
      next = updateTomlTopLevel(next, [
        ["model", codexLocalModelId(spec.model.value)],
      ])
    }
    next = installMagnitudeProvider(next)
    yield* writeIfChanged(paths.codexUser, userSource, next)
    yield* removeConfigurationFile(paths.codex)
    return Option.some({
      model: typeof previousModel === "string"
        ? qualifiedModelSelection(
            typeof previousProvider === "string" ? previousProvider : "openai",
            previousModel,
          )
        : Option.none(),
    })
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const userSource = yield* readOr(paths.codexUser, "")
    const currentProvider = tomlTopLevelValue(userSource, "model_provider")
    const currentModel = tomlTopLevelValue(userSource, "model")
    const currentCatalog = tomlTopLevelValue(userSource, "model_catalog_json")
    let next = removeMagnitudeProvider(userSource)

    if (currentCatalog === paths.codexModels) {
      next = updateTomlTopLevel(next, [["model_catalog_json", undefined]])
    }
    if (currentProvider === CODEX_PROVIDER_ID && isCodexLocalModelId(currentModel) && Option.isSome(spec.restore)) {
      const previous = Option.flatMap(spec.restore.value.model, (selection) =>
        Option.fromNullable(splitQualifiedModelSelection(selection)))
      next = updateTomlTopLevel(next, [
        ["model_provider", Option.match(previous, { onNone: () => undefined, onSome: ({ provider }) => provider })],
        ["model", Option.match(previous, { onNone: () => undefined, onSome: ({ model }) => model })],
      ])
    } else if (currentProvider === CODEX_PROVIDER_ID && !isCodexLocalModelId(currentModel)) {
      // A bundled OpenAI model selected while connected remains selected after
      // removing the proxy provider.
      next = updateTomlTopLevel(next, [["model_provider", "openai"]])
    }
    yield* writeIfChanged(paths.codexUser, userSource, next)
    yield* Effect.forEach([paths.codex, paths.codexModels], removeConfigurationFile, { discard: true })
  }),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", codexLocalModelId(modelId)])
  },
})
