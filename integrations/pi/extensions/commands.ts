import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent"
import { Context, Effect, Exit, Fiber, Layer, ManagedRuntime, Ref, Schema, Scope } from "effect"
import { FetchHttpClient } from "@effect/platform"
import * as NodeCommandExecutor from "@effect/platform-node/NodeCommandExecutor"
import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import {
  MagnitudeClient, MagnitudeServiceStarter, ConnectionErrorSchema, ProtocolMismatch, formatConnectionError,
} from "@magnitudedev/sdk"

class MagnitudeCommandFailed extends Schema.TaggedError<MagnitudeCommandFailed>()("MagnitudeCommandFailed", {
  message: Schema.String,
}) {}
type ModelCommandError = MagnitudeCommandFailed | ProtocolMismatch
const commandFailure = (error: unknown): ModelCommandError => Schema.is(ProtocolMismatch)(error) ? error : new MagnitudeCommandFailed({ message:
  Schema.is(ConnectionErrorSchema)(error) ? formatConnectionError(error) : error instanceof Error ? error.message : String(error),
})
interface ModelCommands {
  readonly stop: Effect.Effect<void, ModelCommandError>
}
const ModelCommands = Context.GenericTag<ModelCommands>("pi/ModelCommands")

const commandsLayer = Layer.effect(ModelCommands, Effect.gen(function* () {
  const client = yield* MagnitudeClient
  return {
    stop: client.models.stop({}).pipe(Effect.asVoid, Effect.mapError(commandFailure)),
  } satisfies ModelCommands
}))

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
