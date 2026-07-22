import { useMemo } from "react"
import { Cause, Duration, Effect, Schedule, Schema, Stream } from "effect"
import { Result, useAtomMount, useAtomValue, type Atom } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import type * as Rpc from "@effect/rpc/Rpc"
import type * as RpcGroup from "@effect/rpc/RpcGroup"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import type {
  MagnitudeRpcs,
  MirroredStateInvalidation,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"

type MagnitudeRpc = RpcGroup.Rpcs<typeof MagnitudeRpcs>
type WatchEvent = MirroredStateInvalidation
type RpcPayload<Tag extends Rpc.Tag<MagnitudeRpc>> = Rpc.PayloadConstructor<Rpc.ExtractTag<MagnitudeRpc, Tag>>
type AgentClientInstance = ReturnType<typeof useAgentClient>

interface ResidentWatch {
  readonly atom: Atom.Atom<unknown>
  readonly mountedMirrorIds: Set<string>
}

const residentWatches = new WeakMap<object, ResidentWatch>()

const runInvalidationWatch = <R>(
  mountedMirrorIds: ReadonlySet<string>,
  connect: Effect.Effect<Stream.Stream<WatchEvent, RpcClientError>, never, R>,
) => {
  const invalidateMountedMirrors = () => Reactivity.invalidate([...mountedMirrorIds])
  const reconnect = Schedule.exponential("100 millis").pipe(
    Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
    Schedule.jittered,
  )
  const watch = Stream.unwrap(Effect.gen(function* () {
    const stream = yield* connect
    yield* Effect.logDebug("Mirrored state watch connected")
    yield* invalidateMountedMirrors()
    return stream.pipe(Stream.tap((event) => Reactivity.invalidate([event.id])))
  }))
  return watch.pipe(
    Stream.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : Effect.logWarning("Mirrored state watch disconnected; retrying").pipe(
        Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 1_000) }),
      )),
    Stream.retry(reconnect),
    Stream.runDrain,
    Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : Effect.logError(Cause.pretty(cause))),
  )
}

const getResidentWatch = (client: AgentClientInstance, mirrorId: string): Atom.Atom<unknown> => {
  const existing = residentWatches.get(client)
  if (existing) {
    existing.mountedMirrorIds.add(mirrorId)
    return existing.atom
  }

  const mountedMirrorIds = new Set([mirrorId])
  const atom = client.runtime.atom(runInvalidationWatch(
    mountedMirrorIds,
    Effect.map(client, (rpc) => rpc("WatchMirroredStates", {})),
  ))
  residentWatches.set(client, { atom, mountedMirrorIds })
  return atom
}

/**
 * Mirrors one protocol-defined backend state into a query atom. The definition's
 * RPC tag is also its invalidation identity, so no parallel client configuration exists.
 */
export function useMirroredState<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
>(definition: {
  readonly id: Id
  readonly getPayload: RpcPayload<Id>
  readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
  readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
}): Result.Result<Snapshot, Error | RpcClientError> {
  const queryAtom = useMirroredStateAtom(definition)
  return useAtomValue(queryAtom)
}

/**
 * Returns the query atom for a mirrored state and keeps the shared invalidation
 * watch resident. Consumers that compose multiple mirrors should derive one
 * atom from these query atoms so React observes a single coherent value.
 */
export function useMirroredStateAtom<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
>(definition: {
  readonly id: Id
  readonly getPayload: RpcPayload<Id>
  readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
  readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
}): Atom.Atom<Result.Result<Snapshot, Error | RpcClientError>> {
  const client = useAgentClient()
  const queryAtom = useMemo(
    () =>
      client.query(definition.id, definition.getPayload, {
        reactivityKeys: [definition.id],
      }),
    [client, definition],
  )
  const watchAtom = useMemo(
    () => getResidentWatch(client, definition.id),
    [client, definition.id],
  )
  useAtomMount(watchAtom)
  return queryAtom
}
