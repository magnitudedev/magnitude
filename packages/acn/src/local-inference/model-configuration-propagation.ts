import { Effect, Layer, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { AgentRuntime } from "../agent-runtime"
import { ProviderClientRegistry } from "../shared-client"
import { LocalModelProviderSource } from "./provider-source"
import { LocalModelConfiguration, type LocalSlotCandidate } from "./model-configuration"

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
  LocalModelConfiguration | ProviderClientRegistry | AgentRuntime | LocalModelProviderSource
> = Layer.scopedDiscard(Effect.gen(function* () {
  const configuration = yield* LocalModelConfiguration
  const providerClients = yield* ProviderClientRegistry
  const agentRuntime = yield* AgentRuntime
  const localSource = yield* LocalModelProviderSource

  yield* propagateModelConfigurationChanges(
    configuration.changes,
    // The local provider source is stable and projects live catalog state;
    // local slot/runtime changes must not rebuild whole provider clients.
    Effect.void,
    agentRuntime.getAllEntries(),
  ).pipe(
    Effect.forkScoped,
  )

  const reconcile = localSource.catalog.list.pipe(
    Effect.provide(FetchHttpClient.layer),
    Effect.map((models) => models.map((model): LocalSlotCandidate => ({
      providerModelId: model.providerModelId,
      availability: model.availability._tag === "Available" ? "available" : "disabled",
      ownership: model.ownership,
      residency: model.residency ?? "unknown",
      productRank: model.productRank ?? Number.MAX_SAFE_INTEGER,
      externalPriority: model.externalPriority ?? Number.MAX_SAFE_INTEGER,
    }))),
    Effect.flatMap(configuration.reconcileSlots),
    Effect.catchAll((cause) => Effect.logWarning("Local model slot reconciliation failed").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )),
  )
  yield* Stream.concat(
    Stream.make(undefined),
    Stream.tick("3 seconds"),
  ).pipe(
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )
}))
