import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent"
import { Context, Effect, Either, Exit, Fiber, Layer, ManagedRuntime, Option, Schema, Scope } from "effect"
import {
  ModelsStatusEnvelopeSchema, ModelsLoadEnvelopeSchema, ModelsStopEnvelopeSchema,
  jsonFailureEnvelopeSchema, type JsonLocalModel, type JsonCommandName,
} from "@magnitudedev/integration-protocol"

class MagnitudeCommandFailed extends Schema.TaggedError<MagnitudeCommandFailed>()("MagnitudeCommandFailed", {
  message: Schema.String,
}) {}
const JsonDocument = Schema.parseJson(Schema.Unknown)
const incompatible = "Magnitude CLI and this extension use incompatible command versions. Update them together."

interface ModelCommands {
  readonly status: (fresh: boolean) => Effect.Effect<typeof ModelsStatusEnvelopeSchema.Type, MagnitudeCommandFailed>
  readonly load: (modelId: string) => Effect.Effect<string, MagnitudeCommandFailed>
  readonly stop: Effect.Effect<void, MagnitudeCommandFailed>
}
const ModelCommands = Context.GenericTag<ModelCommands>("pi/ModelCommands")

const commandsLayer = (pi: ExtensionAPI) => Layer.effect(ModelCommands, Effect.gen(function* () {
  const execute = <A, I>(command: JsonCommandName, args: readonly string[], schema: Schema.Schema<A, I>, timeout: number) =>
    Effect.tryPromise({
      try: (signal) => pi.exec(process.env.MAGNITUDE_CLI?.trim() || "magnitude", [...args, "--json"], { timeout, signal }),
      catch: (error) => new MagnitudeCommandFailed({ message: `Could not execute Magnitude: ${String(error)}` }),
    }).pipe(Effect.flatMap((result) => {
      if (result.killed) return Effect.fail(new MagnitudeCommandFailed({ message: "Magnitude command timed out or was cancelled." }))
      if (result.code !== 0) {
        const failure = Schema.decodeUnknownEither(Schema.parseJson(jsonFailureEnvelopeSchema(command)))(result.stderr)
        const detail = Either.isRight(failure) ? failure.right.error.message : result.stderr.trim() || result.stdout.trim()
        return Effect.fail(new MagnitudeCommandFailed({ message: detail.startsWith("{") || detail.startsWith("[") || /unknown option.*--json/.test(detail)
          ? incompatible : (detail || `Magnitude exited with status ${result.code}`).slice(0, 2_000) }))
      }
      return Schema.decodeUnknown(Schema.parseJson(schema))(result.stdout).pipe(
        Effect.mapError(() => new MagnitudeCommandFailed({ message:
          Either.isLeft(Schema.decodeUnknownEither(JsonDocument)(result.stdout)) ? "Magnitude returned invalid JSON" : incompatible })),
      )
    }))
  const [cached, invalidate] = yield* Effect.cachedInvalidateWithTTL(
    execute("models.status", ["models", "status"], ModelsStatusEnvelopeSchema, 10_000), "30 seconds",
  )
  return {
    status: (fresh) => Effect.gen(function* () {
      if (fresh) yield* invalidate
      const result = yield* cached.pipe(Effect.tapError(() => invalidate))
      if (result.data.state === "initializing") yield* invalidate
      return result
    }),
    load: (modelId) => execute("models.load", ["models", "load", modelId], ModelsLoadEnvelopeSchema, 600_000).pipe(
      Effect.tap(() => invalidate), Effect.map((result) => result.data.modelId),
    ),
    stop: execute("models.stop", ["models", "stop"], ModelsStopEnvelopeSchema, 120_000).pipe(
      Effect.tap(() => invalidate), Effect.asVoid,
    ),
  } satisfies ModelCommands
}))
const stateLabel = (model: JsonLocalModel) => Option.getOrElse(model.residency, () => model.installation)
const loadable = (models: readonly JsonLocalModel[]) => models.filter((model) => model.installation === "installed")

/** Pi callbacks are the sole Promise boundary; runtime disposal cancels subprocess work. */
export const registerMagnitudeCommands = (pi: ExtensionAPI): (() => Promise<void>) => {
  const runtime = ManagedRuntime.make(commandsLayer(pi))
  const scope = Effect.runSync(Scope.make())
  const run = <A, E>(effect: Effect.Effect<A, E, ModelCommands>) => runtime.runPromise(
    Effect.forkIn(effect, scope).pipe(Effect.flatMap(Fiber.join)),
  )
  const invoke = (ctx: ExtensionCommandContext, action: Effect.Effect<void, MagnitudeCommandFailed, ModelCommands>) =>
    run(action.pipe(Effect.catchAll((error) => Effect.sync(() => ctx.ui.notify(error.message, "error")))))
  pi.registerCommand("load-model", {
    description: "Load a Magnitude model",
    getArgumentCompletions: (prefix) => run(Effect.gen(function* () {
      const commands = yield* ModelCommands
      const status = yield* commands.status(false)
      return loadable(status.data.models)
        .filter((model) => model.modelId.toLowerCase().includes(prefix.trim().toLowerCase()))
        .map((model) => ({ value: model.modelId, label: model.displayName, description: stateLabel(model) }))
    }).pipe(Effect.catchAll(() => Effect.succeed(null)))),
    handler: (args, ctx) => invoke(ctx, Effect.gen(function* () {
      const commands = yield* ModelCommands
      let modelId = args.trim()
      if (!modelId) {
        const status = yield* commands.status(true)
        if (status.data.state === "initializing") {
          yield* Effect.sync(() => ctx.ui.notify("Magnitude is discovering local models. Try again shortly.", "info"))
          return
        }
        const eligible = loadable(status.data.models)
        if (!eligible.length) {
          yield* Effect.sync(() => ctx.ui.notify("No installed Magnitude models are available.", "info"))
          return
        }
        const labels = eligible.map((model) => `${model.displayName} · ${stateLabel(model)} · ${model.modelId}`)
        const selected = yield* Effect.promise(() => ctx.ui.select("Load Magnitude model", labels))
        if (selected === undefined) return
        modelId = eligible[labels.indexOf(selected)]?.modelId ?? ""
      }
      if (!modelId) return
      const loaded = yield* commands.load(modelId)
      yield* Effect.sync(() => ctx.ui.notify(`Loaded ${loaded}.`, "info"))
    })),
  })
  pi.registerCommand("stop-model", {
    description: "Stop the active Magnitude model",
    handler: (_args, ctx) => invoke(ctx, Effect.gen(function* () {
      const commands = yield* ModelCommands
      yield* commands.stop
      yield* Effect.sync(() => ctx.ui.notify("Stopped the active Magnitude model.", "info"))
    })),
  })
  return async () => {
    await Effect.runPromise(Scope.close(scope, Exit.void))
    await runtime.dispose()
  }
}
