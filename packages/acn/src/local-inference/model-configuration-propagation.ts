import { Effect, Layer, Stream } from "effect"
import { AgentRuntime } from "../agent-runtime"
import { ProviderClientRegistry } from "../shared-client"
import { LocalModelConfiguration } from "./model-configuration"

export interface ModelConfigurationRefreshEntry {
  readonly session: {
    readonly refreshConfig: () => Effect.Effect<void>
  }
}

export const propagateModelConfigurationChanges = (
  changes: Stream.Stream<void>,
  refreshProviderClients: Effect.Effect<void>,
  getEntries: Effect.Effect<ReadonlyArray<ModelConfigurationRefreshEntry>>,
): Effect.Effect<void> => changes.pipe(
  Stream.runForEach(() => Effect.gen(function* () {
    yield* refreshProviderClients
    const entries = yield* getEntries
    yield* Effect.forEach(entries, (entry) => entry.session.refreshConfig(), {
      concurrency: "unbounded",
      discard: true,
    })
  })),
)

/**
 * Propagates committed model-configuration changes through stable provider
 * clients and already-running agent sessions. Mutation handlers never perform
 * their own session loops.
 */
export const ModelConfigurationPropagationLive: Layer.Layer<
  never,
  never,
  LocalModelConfiguration | ProviderClientRegistry | AgentRuntime
> = Layer.scopedDiscard(Effect.gen(function* () {
  const configuration = yield* LocalModelConfiguration
  const providerClients = yield* ProviderClientRegistry
  const agentRuntime = yield* AgentRuntime

  yield* propagateModelConfigurationChanges(
    configuration.changes,
    providerClients.refreshAll,
    agentRuntime.getAllEntries(),
  ).pipe(
    Effect.forkScoped,
  )
}))
