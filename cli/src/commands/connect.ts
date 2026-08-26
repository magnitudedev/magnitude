import type { Command as Commander } from "@commander-js/extra-typings"
import { Atom, Registry } from "@effect-atom/atom"
import * as FileSystem from "@effect/platform/FileSystem"
import * as PlatformCommand from "@effect/platform/Command"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Client, Mutation } from "@magnitudedev/effect-query"
import type { AgentClient } from "@magnitudedev/client-common"
import {
  MagnitudeBoundary,
  MAGNITUDE_ANTHROPIC_BASE_URL,
  MAGNITUDE_INFERENCE_BASE_URL,
  ProviderModelIdSchema,
  magnitudeImplementationsLayer,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { applyEdits, modify, parse, type ParseError } from "jsonc-parser"
import { homedir } from "node:os"
import { startServer } from "./server"
import { writeFileAtomic } from "../utils/atomic-file"
import { makeTerminalPlatform } from "../platform/terminal"

const INFERENCE_BASE_URL = new URL("v1", MAGNITUDE_INFERENCE_BASE_URL).href.replace(/\/$/, "")
const ANTHROPIC_BASE_URL = MAGNITUDE_ANTHROPIC_BASE_URL
const CLAUDE_GATEWAY_DISCOVERY = "1"
const CLAUDE_GATEWAY_MINIMUM_VERSION = [2, 1, 129] as const
const CLAUDE_CONNECTION_STATE_VERSION = 1
const CODEX_MANAGED_START = "# >>> Magnitude managed provider"
const CODEX_MANAGED_END = "# <<< Magnitude managed provider"

type Harness = "pi" | "opencode" | "codex"

class HarnessConnectError extends Data.TaggedError("HarnessConnectError")<{
  readonly message: string
}> {}

type PriorSetting =
  | { readonly present: false }
  | { readonly present: true; readonly value: unknown }

const PriorSettingSchema = Schema.Union(
  Schema.Struct({ present: Schema.Literal(false) }),
  Schema.Struct({ present: Schema.Literal(true), value: Schema.Unknown }),
)
const ClaudeConnectionStateSchema = Schema.Struct({
  version: Schema.Literal(CLAUDE_CONNECTION_STATE_VERSION),
  settingsFile: Schema.String,
  priorBaseUrl: PriorSettingSchema,
  priorDiscovery: PriorSettingSchema,
})
const ClaudeConnectionStateJson = Schema.parseJson(ClaudeConnectionStateSchema, { space: 2 })

export type ClaudeConnectionState = typeof ClaudeConnectionStateSchema.Type

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

const updateStrictJson = (
  source: string,
  changes: ReadonlyArray<readonly [ReadonlyArray<string>, unknown]>,
): string => {
  strictObject(source)
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

export const updateClaudeConfig = (source: string): string =>
  updateStrictJson(source, [
    [["env", "ANTHROPIC_BASE_URL"], ANTHROPIC_BASE_URL],
    [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
  ])

const claudeSettingsFile = (): string =>
  `${process.env.CLAUDE_CONFIG_DIR ?? `${homedir()}/.claude`}/settings.json`

const claudeConnectionStateFile = (): string =>
  `${homedir()}/.magnitude/integrations/claude-code.json`

const strictObject = (
  source: string,
  scope = "Claude settings.json",
): Record<string, unknown> => {
  const errors: ParseError[] = []
  const value = parse(source, errors, {
    allowTrailingComma: false,
    disallowComments: true,
  })
  if (errors.length > 0 || value === null || typeof value !== "object" || Array.isArray(value)) {
    throw connectFailure(`${scope} is not a strict JSON object`)
  }
  return value as Record<string, unknown>
}

export const validateClaudeCodeVersion = (output: string): string => {
  const match = /(?:^|\s)(\d+)\.(\d+)\.(\d+)(?:\s|$)/.exec(output.trim())
  if (match === null) {
    throw connectFailure(`Unable to parse Claude Code version from: ${output.trim() || "empty output"}`)
  }
  const version = [Number(match[1]), Number(match[2]), Number(match[3])] as const
  let comparison = 0
  for (let index = 0; index < version.length; index += 1) {
    comparison = Math.sign(version[index] - CLAUDE_GATEWAY_MINIMUM_VERSION[index])
    if (comparison !== 0) break
  }
  if (comparison < 0) {
    throw connectFailure(
      `Claude Code ${version.join(".")} does not support gateway model discovery; update to 2.1.129 or later`,
    )
  }
  return version.join(".")
}

const detectClaudeCodeVersion = PlatformCommand.make("claude", "--version").pipe(
  PlatformCommand.string,
  Effect.map((output) => validateClaudeCodeVersion(output)),
  Effect.mapError((error) => error instanceof HarnessConnectError
    ? error
    : connectFailure(`Unable to run claude --version: ${String(error)}`)),
)

const priorSetting = (env: Record<string, unknown>, key: string): PriorSetting =>
  Object.hasOwn(env, key)
    ? { present: true, value: env[key] }
    : { present: false }

export const captureClaudeConnectionState = (
  source: string,
  settingsFile: string,
): ClaudeConnectionState => {
  const value = strictObject(source)
  const env = value.env !== null && typeof value.env === "object" && !Array.isArray(value.env)
    ? value.env as Record<string, unknown>
    : {}
  return {
    version: CLAUDE_CONNECTION_STATE_VERSION,
    settingsFile,
    priorBaseUrl: priorSetting(env, "ANTHROPIC_BASE_URL"),
    priorDiscovery: priorSetting(env, "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"),
  }
}

const parseClaudeConnectionState = (source: string): ClaudeConnectionState | undefined => {
  if (source.trim().length === 0) return undefined
  try {
    return Schema.decodeUnknownSync(ClaudeConnectionStateJson)(source)
  } catch {
    throw connectFailure("Magnitude's Claude Code connection state is invalid")
  }
}

const restoreSetting = (
  changes: Array<readonly [ReadonlyArray<string>, unknown]>,
  env: Record<string, unknown>,
  key: string,
  installed: unknown,
  prior: PriorSetting,
): void => {
  if (env[key] !== installed) return
  changes.push([["env", key], prior.present ? prior.value : undefined])
}

export const restoreClaudeConfig = (
  source: string,
  state: ClaudeConnectionState,
): string => {
  const value = strictObject(source)
  const env = value.env !== null && typeof value.env === "object" && !Array.isArray(value.env)
    ? value.env as Record<string, unknown>
    : {}
  const changes: Array<readonly [ReadonlyArray<string>, unknown]> = []
  restoreSetting(changes, env, "ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL, state.priorBaseUrl)
  restoreSetting(
    changes,
    env,
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
    CLAUDE_GATEWAY_DISCOVERY,
    state.priorDiscovery,
  )
  return updateStrictJson(source, changes)
}

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

const claudeConnectionStatus = (source: string): boolean => {
  const value = strictObject(source)
  const env = value.env !== null && typeof value.env === "object" && !Array.isArray(value.env)
    ? value.env as Record<string, unknown>
    : {}
  return env.ANTHROPIC_BASE_URL === ANTHROPIC_BASE_URL
    && env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY === CLAUDE_GATEWAY_DISCOVERY
}

const validateClaudeConnection = (source: string, scope = "Claude user settings"): void => {
  const value = strictObject(source, scope)
  const env = value.env !== null && typeof value.env === "object" && !Array.isArray(value.env)
    ? value.env as Record<string, unknown>
    : {}
  const base = env.ANTHROPIC_BASE_URL
  if (base !== undefined && base !== ANTHROPIC_BASE_URL) {
    throw connectFailure(`${scope} has a conflicting ANTHROPIC_BASE_URL: ${String(base)}`)
  }
  for (const key of [
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
  ]) {
    if (env[key] === "1" || env[key] === "true") {
      throw connectFailure(`${scope} sets ${key}, which bypasses the Magnitude Anthropic gateway`)
    }
  }
  if (env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC !== undefined) {
    throw connectFailure(
      `${scope} sets CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC, which disables gateway discovery`,
    )
  }
  if (Array.isArray(value.availableModels)) {
    throw connectFailure(`${scope} has an availableModels policy that may exclude gateway models`)
  }
}

export const validateClaudeEnvironment = (
  environment: Readonly<Record<string, string | undefined>>,
): void => {
  const base = environment.ANTHROPIC_BASE_URL
  if (base !== undefined && base !== ANTHROPIC_BASE_URL) {
    throw connectFailure(`The current environment overrides ANTHROPIC_BASE_URL with ${base}`)
  }
  for (const key of [
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
  ]) {
    if (environment[key] === "1" || environment[key] === "true") {
      throw connectFailure(`The current environment sets ${key}, which bypasses the Magnitude gateway`)
    }
  }
  if (environment.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC !== undefined) {
    throw connectFailure(
      "The current environment sets CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC, which disables gateway discovery",
    )
  }
}

const managedClaudeSettingsFile = (): string | undefined => {
  if (process.platform === "darwin") {
    return "/Library/Application Support/ClaudeCode/managed-settings.json"
  }
  if (process.platform === "linux") return "/etc/claude-code/managed-settings.json"
  if (process.platform === "win32") {
    return `${process.env.ProgramFiles ?? "C:\\Program Files"}\\ClaudeCode\\managed-settings.json`
  }
  return undefined
}

const validateHigherPrecedenceClaudeSettings = Effect.gen(function* () {
  const candidates: Array<readonly [string, string]> = [
    [`${process.cwd()}/.claude/settings.json`, "Claude project settings"],
    [`${process.cwd()}/.claude/settings.local.json`, "Claude local project settings"],
  ]
  const managed = managedClaudeSettingsFile()
  if (managed !== undefined) candidates.push([managed, "Claude managed settings"])
  for (const [file, scope] of candidates) {
    const source = yield* readOrEmpty(file)
    if (source.trim().length > 0) validateClaudeConnection(source, `${scope} (${file})`)
  }
})

const discoverClaudeModels = Effect.tryPromise({
  try: async () => {
    const response = await fetch(`${ANTHROPIC_BASE_URL}/v1/models?limit=1000`)
    if (!response.ok) throw new Error(`gateway discovery returned HTTP ${response.status}`)
    const value = await response.json() as {
      readonly data?: unknown
      readonly has_more?: unknown
      readonly first_id?: unknown
      readonly last_id?: unknown
    }
    if (!Array.isArray(value.data) || typeof value.has_more !== "boolean") {
      throw new Error("gateway discovery did not return an Anthropic model-list envelope")
    }
    const models = value.data.map((entry) => {
      if (
        entry === null
        || typeof entry !== "object"
        || (entry as { readonly type?: unknown }).type !== "model"
        || typeof (entry as { readonly id?: unknown }).id !== "string"
      ) {
        throw new Error("gateway discovery returned an invalid model entry")
      }
      return (entry as { readonly id: string }).id
    })
    if (models.some((id) => !id.startsWith("anthropic-local/"))) {
      throw new Error("gateway discovery returned a model outside Magnitude's reserved namespace")
    }
    if (models.length === 0) throw new Error("gateway discovery returned no local models")
    return models
  },
  catch: (error) => connectFailure(`Claude gateway verification failed: ${String(error)}`),
})

const claudeGatewayRunning = Effect.tryPromise({
  try: async () => (await fetch(`${ANTHROPIC_BASE_URL}/api/hello`, { method: "HEAD" })).ok,
  catch: () => false,
}).pipe(Effect.catchAll(() => Effect.succeed(false)))

const connectClaudeCode = (statusOnly: boolean) => Effect.gen(function* () {
  const file = claudeSettingsFile()
  const source = yield* readOrEmptyObject(file)
  validateClaudeEnvironment(process.env)
  validateClaudeConnection(source)
  yield* validateHigherPrecedenceClaudeSettings
  const claudeVersion = yield* detectClaudeCodeVersion
  if (statusOnly) {
    const configured = claudeConnectionStatus(source)
    const running = configured ? yield* claudeGatewayRunning : false
    yield* Effect.sync(() => process.stdout.write(configured
      ? `Claude Code ${claudeVersion} is connected to Magnitude; the gateway is ${running ? "running" : "not running"}.\n`
      : `Claude Code ${claudeVersion} is not connected to Magnitude.\n`))
    return
  }
  yield* startServer
  const models = yield* discoverClaudeModels
  const stateFile = claudeConnectionStateFile()
  const existingState = parseClaudeConnectionState(yield* readOrEmpty(stateFile))
  if (existingState !== undefined && existingState.settingsFile !== file) {
    return yield* connectFailure(
      `Claude configuration moved from ${existingState.settingsFile} to ${file}; disconnect the prior integration first`,
    )
  }
  if (existingState === undefined) {
    const fs = yield* FileSystem.FileSystem
    if (yield* fs.exists(file)) {
      const backup = `${file}.magnitude-backup-${new Date().toISOString().replaceAll(":", "-")}`
      yield* fs.copy(file, backup)
    }
    const state = captureClaudeConnectionState(source, file)
    const stateJson = yield* Schema.encode(ClaudeConnectionStateJson)(state).pipe(
      Effect.mapError((error) => connectFailure(`Unable to encode Claude connection state: ${String(error)}`)),
    )
    yield* writeFileAtomic(
      stateFile,
      `${stateJson.trimEnd()}\n`,
    )
  }
  yield* writeFileAtomic(file, updateClaudeConfig(source))
  yield* Effect.sync(() => process.stdout.write([
    "Magnitude configured Claude Code for gateway discovery.",
    `Claude Code: ${claudeVersion}`,
    `Endpoint: ${ANTHROPIC_BASE_URL}`,
    `Discovered models: ${models.join(", ")}`,
    `Configuration: ${file}`,
    "Run: claude",
    "",
  ].join("\n")))
}).pipe(Effect.mapError((error) => error instanceof HarnessConnectError
  ? error
  : connectFailure(String(error))))

const awaitDownload = (
  client: Pick<AgentClient, "Models">,
  registry: Registry.Registry,
  modelId: ProviderModelId,
  transferObserved = false,
): Effect.Effect<void, unknown> => Registry.getResult(
  registry,
  Atom.make((get) => get(client.Models.GetCatalog({})).result),
).pipe(
  Effect.flatMap((state) => {
    const again = (observed: boolean) => Effect.sleep("250 millis").pipe(
      Effect.zipRight(Effect.suspend(() => awaitDownload(client, registry, modelId, observed))),
    )
    if (state._tag === "Initializing") return again(transferObserved)
    const entry = state.models.find((candidate) =>
      candidate._tag === "Local" && candidate.product.modelId === modelId)
    if (entry?._tag !== "Local") {
      return Effect.fail(connectFailure(`Model ${modelId} is absent from the ACN catalog`))
    }
    switch (entry.product.acquisitionState._tag) {
      case "Installed":
      case "UpdateAvailable":
      case "Updating":
      case "UpdateFailed":
        return Effect.void
      case "Installing":
        return again(true)
      case "InstallFailed":
        return Effect.fail(connectFailure(
          `Model installation failed: ${JSON.stringify(entry.product.acquisitionState.failure)}`,
        ))
      case "NotInstalled":
        // The admitted transfer disappeared without installing: it was
        // cancelled or its failure was acknowledged elsewhere.
        return transferObserved
          ? Effect.fail(connectFailure("Model installation was cancelled"))
          : again(false)
    }
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
        ? awaitDownload(client, registry, providerModelId)
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
    .description("Configure a coding harness to use Magnitude")
    .argument("<harness>", "pi, opencode, codex, or claude-code")
    .argument("[model-id]", "Canonical Magnitude model ID (not used for claude-code)")
    .option("--status", "Verify an existing Claude Code gateway connection")
    .action((harness, modelId, options) => Effect.runPromise((
      harness === "claude-code"
        ? (modelId === undefined
            ? connectClaudeCode(options.status === true).pipe(
                Effect.provide([BunContext.layer, FetchHttpClient.layer]),
              )
            : Effect.fail(connectFailure("claude-code uses dynamic discovery and does not accept a model ID")))
        : (["pi", "opencode", "codex"] as const).includes(harness as Harness)
          && modelId !== undefined
        ? Effect.scoped(connectHarness(harness as Harness, modelId)).pipe(
            Effect.provide([BunContext.layer, FetchHttpClient.layer]),
          )
        : Effect.fail(connectFailure(`Unsupported harness or missing model ID: ${harness}`))
    ).pipe(Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`${error.message}\n`)
      process.exitCode = 1
    })))))

  program.command("disconnect")
    .description("Remove Magnitude-owned Claude Code settings")
    .argument("<harness>", "claude-code")
    .action((harness) => Effect.runPromise(
      harness !== "claude-code"
        ? Effect.sync(() => {
            process.stderr.write(`Unsupported harness: ${harness}\n`)
            process.exitCode = 1
          })
        : Effect.gen(function* () {
            const stateFile = claudeConnectionStateFile()
            const state = parseClaudeConnectionState(yield* readOrEmpty(stateFile))
            if (state === undefined) {
              return yield* connectFailure("Claude Code is not connected by Magnitude")
            }
            const file = state.settingsFile
            yield* writeFileAtomic(
              file,
              restoreClaudeConfig(yield* readOrEmptyObject(file), state),
            )
            const fs = yield* FileSystem.FileSystem
            yield* fs.remove(stateFile)
            yield* Effect.sync(() => {
              process.stdout.write("Claude Code disconnected from Magnitude.\n")
            })
          }).pipe(
            Effect.provide(BunContext.layer),
            Effect.catchAll((error) => Effect.sync(() => {
              process.stderr.write(`${String(error)}\n`)
              process.exitCode = 1
            })),
          ),
    ))
}
