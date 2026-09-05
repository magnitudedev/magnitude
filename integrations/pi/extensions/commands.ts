import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent"
import { Context, Effect, Exit, Fiber, Layer, ManagedRuntime, Match, Option, Ref, Schema, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import {
  MagnitudeClient, MagnitudeServiceStarter, ConnectionErrorSchema, ProtocolMismatch, formatConnectionError,
  ModelIdSchema, formatModelDisplayName, localModelIsInstalled,
  type LocalModel, type ModelCatalogState, type ModelResidency,
} from "@magnitudedev/sdk"

class MagnitudeCommandFailed extends Schema.TaggedError<MagnitudeCommandFailed>()("MagnitudeCommandFailed", {
  message: Schema.String,
}) {}
type ModelCommandError = MagnitudeCommandFailed | ProtocolMismatch
const commandFailure = (error: unknown): ModelCommandError => Schema.is(ProtocolMismatch)(error) ? error : new MagnitudeCommandFailed({ message:
  Schema.is(ConnectionErrorSchema)(error) ? formatConnectionError(error) : error instanceof Error ? error.message : String(error),
})
interface ModelCommands {
  readonly status: (fresh: boolean) => Effect.Effect<ModelCatalogState, ModelCommandError>
  readonly load: (modelId: string) => Effect.Effect<string, ModelCommandError>
  readonly stop: Effect.Effect<void, ModelCommandError>
}
const ModelCommands = Context.GenericTag<ModelCommands>("pi/ModelCommands")

const commandsLayer = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* MagnitudeClient
  const [cached, invalidate] = yield* Effect.cachedInvalidateWithTTL(
    client.models.getCatalog({}).pipe(Effect.mapError(commandFailure)), "30 seconds",
  )
  return {
    status: (fresh) => Effect.gen(function* () {
      if (fresh) yield* invalidate
      const result = yield* cached.pipe(Effect.tapError(() => invalidate))
      if (result._tag === "Initializing") yield* invalidate
      return result
    }),
    load: (input) => Schema.decodeUnknown(ModelIdSchema)(input).pipe(
      Effect.flatMap(modelId => client.models.load({ modelId }).pipe(Effect.as(modelId))),
      Effect.tap(() => invalidate), Effect.mapError(commandFailure),
    ),
    stop: client.models.stop({}).pipe(Effect.tap(() => invalidate), Effect.asVoid, Effect.mapError(commandFailure)),
  } satisfies ModelCommands
}))

const displayName = (model: LocalModel) => formatModelDisplayName(model.presentation.displayName, Option.some(model.presentation.variantLabel))
const residencyLabel = (state: ModelResidency): string => Match.value(state).pipe(Match.tagsExhaustive({
  Unloaded: () => "unloaded",
  Requested: () => "loading",
  Loading: () => "loading",
  Ready: () => "ready",
  Stopping: () => "stopping",
  Failed: () => "failed",
}))
const stateLabel = (model: LocalModel): string => Match.value(model).pipe(Match.tagsExhaustive({
  Discovered: ({ state }) => Match.value(state).pipe(Match.tagsExhaustive({
    Ready: ({ residencyState }) => residencyLabel(residencyState),
    Unavailable: () => "unavailable",
  })),
  Catalog: ({ acquisitionState }) => Match.value(acquisitionState).pipe(Match.tagsExhaustive({
    NotInstalled: () => "not installed",
    Installing: () => "installing",
    InstallFailed: () => "installation failed",
    Installed: ({ residencyState }) => residencyLabel(residencyState),
    UpdateAvailable: ({ residencyState }) => residencyLabel(residencyState),
    Updating: ({ residencyState }) => residencyLabel(residencyState),
    UpdateFailed: ({ residencyState }) => residencyLabel(residencyState),
    Removing: () => "removing",
    RemoveFailed: () => "removal failed",
  })),
}))
const loadable = (catalog: ModelCatalogState): readonly LocalModel[] => catalog._tag === "Initializing" ? [] : catalog.models.flatMap(entry => {
  if (entry._tag !== "Local") return []
  const model = entry.product
  return localModelIsInstalled(model) && (model._tag === "Discovered" ? model.state._tag === "Ready"
    : model.acquisitionState._tag !== "Removing" && model.acquisitionState._tag !== "RemoveFailed") ? [model] : []
}).sort((a,b) => displayName(a).localeCompare(displayName(b)) || a.modelId.localeCompare(b.modelId))

const magnitudeExecutable = () => process.env.MAGNITUDE_CLI?.trim() || "magnitude"
const clientLayer = () => MagnitudeClient.layer().pipe(Layer.provide([
  FetchHttpClient.layer,
  MagnitudeServiceStarter.cliLayer({ executable: magnitudeExecutable() }).pipe(
    Layer.provide(NodeCommandExecutor.layer.pipe(Layer.provide(NodeFileSystem.layer))),
  ),
]))

/** Pi callbacks are the sole Promise boundary; runtime disposal cancels subprocess work. */
export const registerMagnitudeCommands = (pi: ExtensionAPI, sdk: Layer.Layer<MagnitudeClient> = clientLayer()): (() => Promise<void>) => {
  const runtime = ManagedRuntime.make(commandsLayer.pipe(Layer.provide(sdk)))
  const scope = Effect.runSync(Scope.make())
  const repairAttempted = Effect.runSync(Ref.make(false))
  const run = <A, E>(effect: Effect.Effect<A, E, ModelCommands>) => runtime.runPromise(
    Effect.forkIn(effect, scope).pipe(Effect.flatMap(Fiber.join)),
  )
  const invoke = async (ctx: ExtensionCommandContext, action: Effect.Effect<void, ModelCommandError, ModelCommands>) => {
    const reload = await run(action.pipe(
      Effect.as(false),
      Effect.catchTag("ProtocolMismatch", (error) => Effect.gen(function* () {
        if (yield* Ref.getAndSet(repairAttempted, true)) {
          yield* Effect.sync(() => ctx.ui.notify(`${formatConnectionError(error)} Automatic sync was already attempted. Run magnitude connections sync pi and /reload to retry manually.`, "error"))
          return false
        }
        yield* Effect.sync(() => ctx.ui.notify("Magnitude protocol changed. Syncing the Pi plugin with the installed CLI…", "info"))
        const result = yield* Effect.tryPromise({
          try: signal => pi.exec(magnitudeExecutable(), ["connections", "sync", "pi"], { signal, timeout: 120_000 }),
          catch: cause => new MagnitudeCommandFailed({ message: `Magnitude plugin sync failed: ${cause instanceof Error ? cause.message : String(cause)}. Run magnitude connections sync pi manually.` }),
        })
        if (result.killed || result.code !== 0) {
          const detail = result.killed ? "command timed out or was terminated" : result.stderr.trim() || result.stdout.trim() || `exit code ${result.code}`
          return yield* new MagnitudeCommandFailed({ message: `Magnitude plugin sync failed: ${detail.slice(-4_096)}. Check that the installed CLI matches the running service, then run magnitude connections sync pi manually.` })
        }
        yield* Effect.sync(() => ctx.ui.notify("Magnitude plugin synced. Reloading Pi; retry your model command afterward.", "info"))
        return true
      })),
      Effect.catchAll(error => Effect.sync(() => { ctx.ui.notify(error.message, "error"); return false })),
    ))
    // Reload disposes this runtime. Leave its scoped work before calling the host,
    // and never replay the model command or use this context after reload.
    if (reload) await ctx.reload()
  }
  pi.registerCommand("load-model", {
    description: "Load a Magnitude model",
    getArgumentCompletions: (prefix) => run(Effect.gen(function* () {
      const commands = yield* ModelCommands
      const status = yield* commands.status(false)
      return loadable(status)
        .filter((model) => model.modelId.toLowerCase().includes(prefix.trim().toLowerCase()))
        .map((model) => ({ value: model.modelId, label: displayName(model), description: stateLabel(model) }))
    }).pipe(Effect.catchAll(() => Effect.succeed(null)))),
    handler: (args, ctx) => invoke(ctx, Effect.gen(function* () {
      const commands = yield* ModelCommands
      let modelId = args.trim()
      if (!modelId) {
        const status = yield* commands.status(true)
        if (status._tag === "Initializing") {
          yield* Effect.sync(() => ctx.ui.notify("Magnitude is discovering local models. Try again shortly.", "info"))
          return
        }
        const eligible = loadable(status)
        if (!eligible.length) {
          yield* Effect.sync(() => ctx.ui.notify("No installed Magnitude models are available.", "info"))
          return
        }
        const labels = eligible.map((model) => `${displayName(model)} · ${stateLabel(model)} · ${model.modelId}`)
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
