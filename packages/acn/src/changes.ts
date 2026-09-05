import { Context, Effect, Layer, Queue, Ref, Stream } from "effect"
import {
  Projects,
  Sessions,
  type Change,
} from "@magnitudedev/acn-protocol"
import { ProjectStore } from "./project-store"
import { SessionInspector } from "./session-inspector"

/**
 * The ACN change registry: every change source publishes pokes in the
 * clients' operation-identity space, and `StreamChanges` serves the multiplexed
 * stream. Each connected subscriber retains bounded, operation-keyed invalidations;
 * repeated keys coalesce and excessive keyed entries broaden to one whole-operation
 * invalidation. Reconnect rereads authoritative state instead of replaying history.
 */
export interface AcnChangesApi {
  readonly publish: (change: Change) => Effect.Effect<void>
  readonly stream: Stream.Stream<Change>
}

export class AcnChanges extends Context.Tag("AcnChanges")<AcnChanges, AcnChangesApi>() {}

const MAX_PENDING_KEYS_PER_QUERY = 64

interface PendingQueryInvalidation {
  readonly all: boolean
  readonly keyed: ReadonlyMap<string, Change>
}

interface ChangeSubscriber {
  readonly offer: (change: Change) => Effect.Effect<void>
  readonly stream: Stream.Stream<Change>
  readonly shutdown: Effect.Effect<void>
}

const keyOf = (change: Change): string => JSON.stringify(change.key)

const addPending = (
  pending: ReadonlyMap<string, PendingQueryInvalidation>,
  change: Change,
): ReadonlyMap<string, PendingQueryInvalidation> => {
  const existing = pending.get(change.operation)
  if (existing?.all === true) return pending
  const next = new Map(pending)
  if (change.key === undefined) {
    next.set(change.operation, { all: true, keyed: new Map() })
    return next
  }
  const keyed = new Map(existing?.keyed)
  keyed.set(keyOf(change), change)
  next.set(change.operation, keyed.size > MAX_PENDING_KEYS_PER_QUERY
    ? { all: true, keyed: new Map() }
    : { all: false, keyed })
  return next
}

const makeSubscriber = Effect.gen(function* () {
  const pending = yield* Ref.make<ReadonlyMap<string, PendingQueryInvalidation>>(new Map())
  const available = yield* Queue.sliding<void>(1)
  const take = Ref.getAndSet(pending, new Map()).pipe(
    Effect.map((queries) => [...queries].flatMap(([operation, invalidation]) =>
      invalidation.all ? [{ operation }] : [...invalidation.keyed.values()])),
  )
  return {
    offer: (change: Change) => Ref.update(pending, (current) => addPending(current, change)).pipe(
      Effect.zipRight(Queue.offer(available, undefined)),
      Effect.asVoid,
    ),
    stream: Stream.fromQueue(available).pipe(
      Stream.mapEffect(() => take),
      Stream.flatMap(Stream.fromIterable),
    ),
    shutdown: Queue.shutdown(available),
  } satisfies ChangeSubscriber
})

export const AcnChangesLive: Layer.Layer<AcnChanges> = Layer.effect(
  AcnChanges,
  Effect.gen(function* () {
    const subscribers = yield* Ref.make<ReadonlySet<ChangeSubscriber>>(new Set())
    const stream = Stream.unwrapScoped(Effect.acquireRelease(
      makeSubscriber.pipe(Effect.tap((subscriber) => Ref.update(subscribers, (current) => {
        const next = new Set(current)
        next.add(subscriber)
        return next
      }))),
      (subscriber) => Ref.update(subscribers, (current) => {
        const next = new Set(current)
        next.delete(subscriber)
        return next
      }).pipe(Effect.zipRight(subscriber.shutdown)),
    ).pipe(Effect.map((subscriber) => subscriber.stream)))
    return AcnChanges.of({
      publish: (change) => Ref.get(subscribers).pipe(
        Effect.flatMap((current) => Effect.forEach(
          current,
          (subscriber) => subscriber.offer(change),
          { concurrency: "unbounded", discard: true },
        )),
      ),
      stream,
    })
  }),
)

/** Queries whose authoritative data a project-store commit may change. */
export const projectChangeQueries: ReadonlyArray<string> = [Projects.listProjects._tag, Projects.inspectProject._tag]
/** Queries whose authoritative data a session-metadata commit may change. */
export const sessionChangeQueries: ReadonlyArray<string> = [
  Sessions.listSessions._tag,
  Sessions.listRecentSessionDirectories._tag,
  Sessions.getSession._tag,
]

/**
 * Forwards storage-level change streams into the registry: a store commit
 * names every operation it backs. Versioned snapshots publish their own pokes.
 */
export const AcnStorageChangesLive: Layer.Layer<never, never, AcnChanges | ProjectStore | SessionInspector> =
  Layer.scopedDiscard(Effect.gen(function* () {
    const changes = yield* AcnChanges
    const projects = yield* ProjectStore
    const sessions = yield* SessionInspector
    const forward = (source: Stream.Stream<unknown>, queries: ReadonlyArray<string>) =>
      source.pipe(
        Stream.runForEach(() => Effect.forEach(queries, (operation) => changes.publish({ operation }), { discard: true })),
        Effect.forkScoped,
      )
    yield* forward(projects.changes, projectChangeQueries)
    yield* forward(sessions.changes, sessionChangeQueries)
  }))
