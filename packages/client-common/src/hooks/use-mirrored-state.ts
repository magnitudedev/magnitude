import { useMemo } from "react"
import { Cause, Duration, Effect, Equivalence, Option, Schedule, Schema, Stream } from "effect"
import { Atom, Result, useAtomMount, useAtomValue } from "@effect-atom/atom-react"
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
  readonly atom: Atom.Atom<Result.Result<void, never>>
  readonly mountedMirrorIds: Set<string>
  readonly subscribers: Map<string, Set<() => Effect.Effect<void>>>
}

const residentWatches = new WeakMap<object, ResidentWatch>()

const runInvalidationWatch = <R>(
  mountedMirrorIds: ReadonlySet<string>,
  subscribers: ReadonlyMap<string, ReadonlySet<() => Effect.Effect<void>>>,
  connect: Effect.Effect<Stream.Stream<WatchEvent, RpcClientError>, never, R>,
) => {
  const notify = (ids: ReadonlyArray<string>) => Effect.gen(function* () {
    yield* Reactivity.invalidate(ids)
    yield* Effect.forEach(
      ids.flatMap((id) => [...(subscribers.get(id) ?? [])]),
      (subscriber) => subscriber(),
      { discard: true },
    )
  })
  const reconnect = Schedule.exponential("100 millis").pipe(
    Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
    Schedule.jittered,
  )
  const watch = Stream.unwrap(Effect.gen(function* () {
    const stream = yield* connect
    yield* Effect.logDebug("Mirrored state watch connected")
    yield* notify([...mountedMirrorIds])
    return stream.pipe(Stream.tap((event) => notify([event.id])))
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

export const getMirroredStateInvalidationWatch = (
  client: AgentClientInstance,
  mirrorId: string,
): Atom.Atom<Result.Result<void, never>> => {
  const existing = residentWatches.get(client)
  if (existing) {
    existing.mountedMirrorIds.add(mirrorId)
    return existing.atom
  }

  const mountedMirrorIds = new Set([mirrorId])
  const subscribers = new Map<string, Set<() => Effect.Effect<void>>>()
  const atom = client.runtime.atom(runInvalidationWatch(
    mountedMirrorIds,
    subscribers,
    Effect.map(client, (rpc) => rpc("WatchMirroredStates", {})),
  ))
  residentWatches.set(client, { atom, mountedMirrorIds, subscribers })
  return atom
}

export const subscribeToMirroredStateInvalidation = (
  client: AgentClientInstance,
  mirrorId: string,
  subscriber: () => Effect.Effect<void>,
): (() => void) => {
  getMirroredStateInvalidationWatch(client, mirrorId)
  const resident = residentWatches.get(client)!
  const current = resident.subscribers.get(mirrorId) ?? new Set()
  current.add(subscriber)
  resident.subscribers.set(mirrorId, current)
  return () => {
    current.delete(subscriber)
    if (current.size === 0) resident.subscribers.delete(mirrorId)
  }
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
 * Selects a semantically stable value from one mirrored-state query.
 *
 * The selector remains a read-only projection of the canonical query atom. Its
 * retained value is only an equality cache: when a fresh snapshot projects to
 * an equivalent value, consumers keep the previous reference and do not
 * rerender. No server state is copied into a writable client atom.
 */
export function useMirroredStateSelector<
  const Id extends Rpc.Tag<MagnitudeRpc>,
  Snapshot,
  SnapshotEncoded,
  SnapshotRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
  Selection,
>(
  definition: {
    readonly id: Id
    readonly getPayload: RpcPayload<Id>
    readonly snapshotSchema: Schema.Schema<Snapshot, SnapshotEncoded, SnapshotRequirements>
    readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
  },
  selector: (snapshot: Snapshot) => Selection,
  equivalent: Equivalence.Equivalence<Selection>,
): Option.Option<Selection> {
  const queryAtom = useMirroredStateAtom(definition)
  const selectorAtom = useMemo(() => {
    let previousSelection = Option.none<Selection>()
    return Atom.make((get) => Option.map(
      Result.value(get(queryAtom)),
      (snapshot) => {
        const nextSelection = selector(snapshot)
        if (Option.isSome(previousSelection)
          && equivalent(previousSelection.value, nextSelection)) {
          return previousSelection.value
        }
        previousSelection = Option.some(nextSelection)
        return nextSelection
      },
    ))
  }, [equivalent, queryAtom, selector])
  return useAtomValue(selectorAtom)
}

/**
 * Returns the query atom for one mirrored domain and keeps the shared
 * invalidation watch resident. Consumers must preserve each domain's Result;
 * successful values may be composed purely at the rendering boundary.
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
        timeToLive: Infinity,
      }),
    [client, definition],
  )
  const watchAtom = useMemo(
    () => getMirroredStateInvalidationWatch(client, definition.id),
    [client, definition.id],
  )
  useAtomMount(watchAtom)
  return queryAtom
}
