import type { Command as Commander } from "@commander-js/extra-typings"
import { Atom, Registry } from "@effect-atom/atom"
import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Client, Mutation } from "@magnitudedev/effect-query"
import type { AgentClient } from "@magnitudedev/client-common"
import {
  MagnitudeBoundary,
  MAGNITUDE_INFERENCE_BASE_URL,
  ProviderModelIdSchema,
  magnitudeImplementationsLayer,
  type ModelDownloadId,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { applyEdits, modify, parse, type ParseError } from "jsonc-parser"
import { homedir } from "node:os"
import { startServer } from "./server"
import { writeFileAtomic } from "../utils/atomic-file"
import { makeTerminalPlatform } from "../platform/terminal"

const INFERENCE_BASE_URL = new URL("v1", MAGNITUDE_INFERENCE_BASE_URL).href.replace(/\/$/, "")
const CODEX_MANAGED_START = "# >>> Magnitude managed provider"
const CODEX_MANAGED_END = "# <<< Magnitude managed provider"

type Harness = "pi" | "opencode" | "codex"

class HarnessConnectError extends Data.TaggedError("HarnessConnectError")<{
  readonly message: string
}> {}

const connectFailure = (message: string) => new HarnessConnectError({ message })
const tomlString = (value: string): string => JSON.stringify(value)

export const renderCodexMagnitudeBlock = (modelId: string): string => `${CODEX_MANAGED_START}
[model_providers.magnitude]
name = "Magnitude"
base_url = ${tomlString(INFERENCE_BASE_URL)}
wire_api = "chat"
requires_openai_auth = false

[profiles.magnitude]
model_provider = "magnitude"
model = ${tomlString(modelId)}
${CODEX_MANAGED_END}`

export const updateCodexConfig = (
  source: string,
  modelId: string,
): string => {
  const block = renderCodexMagnitudeBlock(modelId)
  const start = source.indexOf(CODEX_MANAGED_START)
  const end = source.indexOf(CODEX_MANAGED_END)
  if (start >= 0 || end >= 0) {
    if (start < 0 || end < start) {
      throw connectFailure("The existing Magnitude block in Codex config.toml is malformed")
    }
    return `${source.slice(0, start)}${block}${source.slice(end + CODEX_MANAGED_END.length)}`
  }
  if (/^\s*\[(?:model_providers|profiles)\.magnitude\]\s*$/m.test(source)) {
    throw connectFailure("Codex config.toml already contains a non-managed Magnitude provider or profile")
  }
  return `${source.trimEnd()}${source.trim().length === 0 ? "" : "\n\n"}${block}\n`
}

const updateJsonc = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string>, unknown]>,
): string => {
  const errors: ParseError[] = []
  const parsed = parse(source, errors, { allowTrailingComma: true, disallowComments: false })
  if (errors.length > 0 || typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw connectFailure("The harness configuration is not a valid JSON object")
  }
  return changes.reduce((current, [path, value]) => applyEdits(
    current,
    modify(current, [...path], value, {
      formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" },
    }),
  ), source)
}

export const updateOpenCodeConfig = (source: string, modelId: string): string => updateJsonc(source, [
  [["provider", "magnitude"], {
    npm: "@ai-sdk/openai-compatible",
    name: "Magnitude",
    options: { baseURL: INFERENCE_BASE_URL, apiKey: "magnitude" },
    models: { [modelId]: { name: modelId } },
  }],
  [["model"], `magnitude/${modelId}`],
])

export const updatePiModelsConfig = (source: string, modelId: string): string => updateJsonc(source, [[
  ["providers", "magnitude"],
  {
    baseUrl: INFERENCE_BASE_URL,
    api: "openai-completions",
    apiKey: "magnitude",
    models: [{ id: modelId }],
  },
]])

export const updatePiSettingsConfig = (source: string, modelId: string): string => updateJsonc(source, [
  [["defaultProvider"], "magnitude"],
  [["defaultModel"], modelId],
])

const readOrEmptyObject = (file: string) => FileSystem.FileSystem.pipe(
  Effect.flatMap((fs) => fs.readFileString(file)),
  Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
    ? Effect.succeed("{}\n")
    : Effect.fail(error)),
)

const readOrEmpty = (file: string) => FileSystem.FileSystem.pipe(
  Effect.flatMap((fs) => fs.readFileString(file)),
  Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
    ? Effect.succeed("")
    : Effect.fail(error)),
)

const configureHarness = (harness: Harness, modelId: string) => Effect.gen(function* () {
  if (harness === "opencode") {
    const file = `${homedir()}/.config/opencode/opencode.json`
    yield* writeFileAtomic(file, updateOpenCodeConfig(yield* readOrEmptyObject(file), modelId))
    return { file, invocation: "opencode" }
  }
  if (harness === "pi") {
    const modelsFile = `${homedir()}/.pi/agent/models.json`
    const settingsFile = `${homedir()}/.pi/agent/settings.json`
    yield* writeFileAtomic(modelsFile, updatePiModelsConfig(yield* readOrEmptyObject(modelsFile), modelId))
    yield* writeFileAtomic(settingsFile, updatePiSettingsConfig(yield* readOrEmptyObject(settingsFile), modelId))
    return { file: `${modelsFile}, ${settingsFile}`, invocation: "pi" }
  }
  const file = `${homedir()}/.codex/config.toml`
  yield* writeFileAtomic(file, updateCodexConfig(yield* readOrEmpty(file), modelId))
  return { file, invocation: "codex --profile magnitude" }
}).pipe(Effect.mapError((error) => error instanceof HarnessConnectError
  ? error
  : connectFailure(String(error))))

const awaitDownload = (
  client: Pick<AgentClient, "Models">,
  registry: Registry.Registry,
  modelId: ProviderModelId,
  downloadId: ModelDownloadId,
): Effect.Effect<void, unknown> => Registry.getResult(
  registry,
  Atom.make((get) => get(client.Models.GetCatalog({})).result),
).pipe(
  Effect.flatMap((state) => {
    if (state._tag === "Initializing") return Effect.sleep("250 millis").pipe(
      Effect.zipRight(Effect.suspend(() => awaitDownload(client, registry, modelId, downloadId))),
    )
    const entry = state.models.find((candidate) =>
      candidate._tag === "Local" && candidate.product.modelId === modelId)
    if (entry?._tag !== "Local") {
      return Effect.fail(connectFailure(`Model ${modelId} is absent from the ACN catalog`))
    }
    const acquisition = entry.product.acquisitionState
    if (acquisition._tag === "Installed") return Effect.void
    if (acquisition._tag === "Failed" && acquisition.downloadId === downloadId) {
      return Effect.fail(connectFailure(`Model installation failed: ${JSON.stringify(acquisition.failure)}`))
    }
    if (acquisition._tag === "Cancelled" && acquisition.downloadId === downloadId) {
      return Effect.fail(connectFailure("Model installation was cancelled"))
    }
    return Effect.sleep("250 millis").pipe(
      Effect.zipRight(Effect.suspend(() => awaitDownload(client, registry, modelId, downloadId))),
    )
  }),
)

const connectHarness = (harness: Harness, modelId: string) => Effect.gen(function* () {
  yield* startServer
  const terminal = yield* makeTerminalPlatform({
    launchCommand: Option.none(),
    debug: false,
    effectLoggingLayer: Option.none(),
  })
  const client = Client.make(
    MagnitudeBoundary,
    magnitudeImplementationsLayer(terminal.platform.protocolLayer),
  )
  const registry = Registry.make()
  yield* Effect.addFinalizer(() => Effect.sync(() => registry.dispose()))
  const providerModelId = yield* Schema.decodeUnknown(ProviderModelIdSchema)(modelId)
  const configured = yield* Mutation.execute(
    client.Models.InstallLocalModel,
    { modelId: providerModelId },
  ).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
      Effect.flatMap((admission) => admission._tag === "DownloadAdmitted"
        ? awaitDownload(client, registry, providerModelId, admission.downloadId)
        : Effect.void),
      Effect.zipRight(configureHarness(harness, modelId)),
  )
  yield* Effect.sync(() => process.stdout.write([
    `Magnitude configured ${harness} for ${modelId}.`,
    `Configuration: ${configured.file}`,
    `Run: ${configured.invocation}`,
    "",
  ].join("\n")))
}).pipe(Effect.mapError((error) => error instanceof HarnessConnectError
  ? error
  : connectFailure(String(error))))

export const registerConnectCommand = (program: Commander): void => {
  program.command("connect")
    .description("Install a model and configure a coding harness to use Magnitude")
    .argument("<harness>", "pi, opencode, or codex")
    .argument("<model-id>", "Canonical Magnitude model ID")
    .action((harness, modelId) => Effect.runPromise(
      (["pi", "opencode", "codex"] as const).includes(harness as Harness)
        ? Effect.scoped(connectHarness(harness as Harness, modelId)).pipe(
            Effect.provide([BunContext.layer, FetchHttpClient.layer]),
            Effect.catchAll((error) => Effect.sync(() => {
              process.stderr.write(`${error.message}\n`)
              process.exitCode = 1
            })),
          )
        : Effect.sync(() => {
            process.stderr.write(`Unsupported harness: ${harness}\n`)
            process.exitCode = 1
          }),
    ))
}
