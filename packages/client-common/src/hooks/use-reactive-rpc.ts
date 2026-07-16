import { useMemo } from "react"
import { Cause, Duration, Effect, Schedule, Stream } from "effect"
import { Result, useAtomMount, useAtomValue, type Atom } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import type { MirroredResourceInvalidation, StreamHeartbeat } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { AgentClient } from "../state/agent-client"
import { makeResidentWatchRegistry } from "../state/reactive-rpc-watch-registry"

export type ReactiveRpcResource = "modelCatalog" | "modelSlots" | "localInference"

type WatchEvent = MirroredResourceInvalidation | StreamHeartbeat

const residentWatches = makeResidentWatchRegistry<ReactiveRpcResource>()

const runInvalidationWatch = <E, R>(
  resource: ReactiveRpcResource,
  connect: Effect.Effect<Stream.Stream<WatchEvent, E>, never, R>,
) => {
  const reconnect = Schedule.exponential("100 millis").pipe(
    Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
    Schedule.jittered,
  )
  const watch = Stream.unwrap(Effect.gen(function* () {
    const stream = yield* connect
    yield* Effect.logDebug("Reactive resource watch connected").pipe(
      Effect.annotateLogs({ resource }),
    )
    yield* Reactivity.invalidate([resource])
    return stream.pipe(
      Stream.filter((event) => event._tag === "changed"),
      Stream.tap(() => Reactivity.invalidate([resource])),
    )
  }))
  return watch.pipe(
    Stream.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : Effect.logWarning("Reactive resource watch disconnected; retrying").pipe(
        Effect.annotateLogs({ resource, cause: Cause.pretty(cause).slice(0, 1_000) }),
      )),
    Stream.retry(reconnect),
    Stream.runDrain,
    Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : Effect.logError(`[${resource}] ${Cause.pretty(cause)}`)),
  )
}

function useQueryWithWatch<A, E>(
  queryAtom: Atom.Atom<Result.Result<A, E>>,
  watchAtom: Atom.Atom<unknown>,
): Result.Result<A, E> {
  useAtomMount(watchAtom)
  return useAtomValue(queryAtom)
}

type AgentClientInstance = ReturnType<typeof useAgentClient>

interface MirroredResourceConfig<A, E> {
  readonly resource: ReactiveRpcResource
  readonly query: (client: AgentClientInstance) => Atom.Atom<Result.Result<A, E>>
  readonly watch: (client: AgentClientInstance) => Effect.Effect<Stream.Stream<WatchEvent, E>, never, AgentClient>
}

/** Query-backed authoritative resource mirrored through an invalidation-only watch. */
export function useMirroredResource<A, E>(config: MirroredResourceConfig<A, E>): Result.Result<A, E> {
  const client = useAgentClient()
  const queryAtom = useMemo(
    () => config.query(client),
    [client, config],
  )
  const watchAtom = useMemo(
    () => residentWatches.getOrCreate(client, config.resource, () => client.runtime.atom(runInvalidationWatch(
      config.resource,
      config.watch(client),
    ))),
    [client, config],
  )
  return useQueryWithWatch(queryAtom, watchAtom)
}

const modelCatalogResource = {
  resource: "modelCatalog",
  query: (client: AgentClientInstance) => client.query("GetModelCatalog", {}, { reactivityKeys: ["modelCatalog"] }),
  watch: (client: AgentClientInstance) => Effect.map(client, (rpc) => rpc("WatchModelCatalog", {})),
} as const

const modelSlotsResource = {
  resource: "modelSlots",
  query: (client: AgentClientInstance) => client.query("GetModelSlots", {}, { reactivityKeys: ["modelSlots"] }),
  watch: (client: AgentClientInstance) => Effect.map(client, (rpc) => rpc("WatchModelSlots", {})),
} as const

const localInferenceResource = {
  resource: "localInference",
  query: (client: AgentClientInstance) => client.query("GetLocalInferenceState", {}, { reactivityKeys: ["localInference"] }),
  watch: (client: AgentClientInstance) => Effect.map(client, (rpc) => rpc("WatchLocalInferenceState", {})),
} as const

export const useModelCatalog = () => useMirroredResource(modelCatalogResource)

export const useModelSlots = () => useMirroredResource(modelSlotsResource)

export const useLocalInferenceResource = () => useMirroredResource(localInferenceResource)
