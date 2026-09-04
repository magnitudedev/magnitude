import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent"
import { Either, Option, Schema } from "effect"

const COMMAND_TIMEOUT_MS = 120_000
const INCOMPATIBLE_CLI_MESSAGE = "This Magnitude extension requires a newer Magnitude CLI."
const magnitudeExecutable = (): string => process.env.MAGNITUDE_CLI ?? "magnitude"

const JsonModelSchema = Schema.Struct({
  modelId: Schema.NonEmptyString,
  displayName: Schema.NonEmptyString,
  installation: Schema.Literal("not_installed", "installing", "installed", "removing", "unavailable"),
  residency: Schema.optionalWith(
    Schema.Literal("unloaded", "loading", "ready", "stopping", "failed"),
    { as: "Option", exact: true },
  ),
})
type JsonModel = typeof JsonModelSchema.Type

const StatusEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.status"),
  ok: Schema.Literal(true),
  data: Schema.Union(
    Schema.Struct({ state: Schema.Literal("initializing"), models: Schema.Tuple() }),
    Schema.Struct({ state: Schema.Literal("ready"), models: Schema.Array(JsonModelSchema) }),
  ),
})

const LoadEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.load"),
  ok: Schema.Literal(true),
  data: Schema.Struct({ modelId: Schema.NonEmptyString }),
})

const StopEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.stop"),
  ok: Schema.Literal(true),
  data: Schema.Struct({}),
})

const FailureEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.status", "models.load", "models.stop"),
  ok: Schema.Literal(false),
  error: Schema.Struct({ message: Schema.String }),
})
const JsonDocumentSchema = Schema.parseJson(Schema.Unknown)

const parseFailureMessage = (source: string, expectedCommand: string): string | undefined => {
  const json = Schema.decodeUnknownEither(JsonDocumentSchema)(source)
  if (Either.isLeft(json)) return undefined
  const decoded = Schema.decodeUnknownEither(FailureEnvelopeSchema)(json.right)
  return Either.isRight(decoded) && decoded.right.command === expectedCommand
    ? decoded.right.error.message
    : undefined
}

const executeJson = async <A, I>(
  pi: ExtensionAPI,
  arguments_: ReadonlyArray<string>,
  schema: Schema.Schema<A, I>,
): Promise<A> => {
  const result = await pi.exec(magnitudeExecutable(), [...arguments_, "--json"], { timeout: COMMAND_TIMEOUT_MS })
  if (result.code !== 0) {
    const expectedCommand = arguments_.slice(0, 2).join(".")
    const stderr = result.stderr.trim()
    const stdout = result.stdout.trim()
    const message = parseFailureMessage(stderr, expectedCommand)
    if (message !== undefined) throw new Error(message)
    const fallback = stderr || stdout
    if (fallback.includes("unknown option '--json'") || fallback.includes('unknown option "--json"')) {
      throw new Error(INCOMPATIBLE_CLI_MESSAGE)
    }
    if (fallback.startsWith("{") || fallback.startsWith("[")) {
      throw new Error("Magnitude returned an incompatible JSON error response")
    }
    throw new Error(fallback || `Magnitude exited with status ${result.code}`)
  }
  const json = Schema.decodeUnknownEither(JsonDocumentSchema)(result.stdout)
  if (Either.isLeft(json)) throw new Error("Magnitude returned invalid JSON")
  const decoded = Schema.decodeUnknownEither(schema)(json.right)
  if (Either.isLeft(decoded)) throw new Error(INCOMPATIBLE_CLI_MESSAGE)
  return decoded.right
}

const listModels = async (pi: ExtensionAPI): Promise<ReadonlyArray<JsonModel>> => {
  const envelope = await executeJson(pi, ["models", "status"], StatusEnvelopeSchema)
  if (envelope.data.state === "initializing") return []
  return envelope.data.models
}

const modelState = (model: JsonModel): string => Option.match(model.residency, {
  onNone: () => model.installation,
  onSome: (state) => state,
})
const modelLabel = (model: JsonModel): string => `${model.displayName} · ${modelState(model)} · ${model.modelId}`
const loadableModels = (models: ReadonlyArray<JsonModel>): ReadonlyArray<JsonModel> => models.filter(
  (model) => model.installation === "installed",
)

const reportError = (ctx: ExtensionCommandContext, error: unknown): void => {
  ctx.ui.notify(error instanceof Error ? error.message : String(error), "error")
}

export const registerMagnitudeCommands = (pi: ExtensionAPI): void => {
  let cachedModels: { readonly expiresAt: number; readonly models: ReadonlyArray<JsonModel> } | undefined
  const cachedList = async (): Promise<ReadonlyArray<JsonModel>> => {
    if (cachedModels !== undefined && cachedModels.expiresAt > Date.now()) return cachedModels.models
    const models = await listModels(pi)
    cachedModels = { expiresAt: Date.now() + 30_000, models }
    return models
  }
  const completions = async (prefix: string) => {
    try {
      return loadableModels(await cachedList())
        .filter((model) => model.modelId.toLowerCase().includes(prefix.trim().toLowerCase()))
        .map((model) => ({ value: model.modelId, label: model.displayName, description: modelState(model) }))
    } catch {
      return null
    }
  }

  pi.registerCommand("load-model", {
    description: "Load a Magnitude model",
    getArgumentCompletions: completions,
    handler: async (args, ctx) => {
      try {
        let modelId = args.trim()
        if (modelId.length === 0) {
          const eligible = loadableModels(await cachedList())
          if (eligible.length === 0) {
            ctx.ui.notify("No installed Magnitude models are available.", "info")
            return
          }
          const labels = eligible.map(modelLabel)
          const selected = await ctx.ui.select("Load Magnitude model", labels)
          if (selected === undefined) return
          modelId = eligible[labels.indexOf(selected)]?.modelId ?? ""
        }
        if (modelId.length === 0) return
        const result = await executeJson(pi, ["models", "load", modelId], LoadEnvelopeSchema)
        cachedModels = undefined
        ctx.ui.notify(`Loaded ${result.data.modelId}.`, "info")
      } catch (error) {
        reportError(ctx, error)
      }
    },
  })

  pi.registerCommand("stop-model", {
    description: "Stop the active Magnitude model",
    handler: async (_args, ctx) => {
      try {
        await executeJson(pi, ["models", "stop"], StopEnvelopeSchema)
        cachedModels = undefined
        ctx.ui.notify("Stopped the active Magnitude model.", "info")
      } catch (error) {
        reportError(ctx, error)
      }
    },
  })
}
