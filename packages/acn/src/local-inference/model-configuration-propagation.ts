import { Effect, Layer, Stream } from "effect"
import { LocalModelProviderSource } from "./provider-source"
import { LocalModelConfiguration } from "./model-configuration"

/**
 * Reconciles local runtime readiness into authoritative slot configuration.
 * Account publishes resulting coherent configuration snapshots; sessions
 * subscribe directly and therefore require no per-session propagation loop.
 */
export const ModelConfigurationPropagationLive: Layer.Layer<
  never,
  never,
  LocalModelConfiguration | LocalModelProviderSource
> = Layer.scopedDiscard(Effect.gen(function* () {
  const configuration = yield* LocalModelConfiguration
  const localSource = yield* LocalModelProviderSource

  const reconcile = localSource.selectionReady.pipe(
    Effect.flatMap((ready) => ready
      ? localSource.selectionInput.pipe(Effect.flatMap(configuration.reconcileSlots))
      : Effect.succeed(false)),
    Effect.catchAll((cause) => Effect.logWarning("Local model slot reconciliation failed").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  )
  yield* Stream.concat(Stream.make(undefined), localSource.selectionChanges).pipe(
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )
}))
