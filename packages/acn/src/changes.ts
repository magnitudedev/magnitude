import { Context, Effect, Layer, PubSub, Stream } from "effect"
import {
  Projects,
  Sessions,
  type Change,
} from "@magnitudedev/acn-protocol"
import { ProjectStore } from "./project-store"
import { SessionInspector } from "./session-inspector"

/**
 * The ACN change registry: every change source publishes pokes in the
 * clients' query-identity space, and `StreamChanges` serves the multiplexed
 * stream. Publishing is fire-and-forget; the stream is bounded and coalescing,
 * so a late subscriber rereads authoritative state instead of replaying history.
 */
export interface AcnChangesApi {
  readonly publish: (change: Change) => Effect.Effect<void>
  readonly stream: Stream.Stream<Change>
}

export class AcnChanges extends Context.Tag("AcnChanges")<AcnChanges, AcnChangesApi>() {}

export const AcnChangesLive: Layer.Layer<AcnChanges> = Layer.effect(
  AcnChanges,
  Effect.gen(function* () {
    const events = yield* PubSub.sliding<Change>(256)
    return AcnChanges.of({
      publish: (change) => PubSub.publish(events, change).pipe(Effect.asVoid),
      stream: Stream.fromPubSub(events),
    })
  }),
)

/** Queries whose authoritative data a project-store commit may change. */
export const projectChangeQueries: ReadonlyArray<string> = [Projects.ListProjects.name, Projects.InspectProject.name]
/** Queries whose authoritative data a session-metadata commit may change. */
export const sessionChangeQueries: ReadonlyArray<string> = [
  Sessions.ListSessions.name,
  Sessions.ListRecentSessionDirectories.name,
  Sessions.GetSession.name,
]

/**
 * Forwards storage-level change streams into the registry: a store commit
 * names every query it backs. Versioned snapshots publish their own pokes.
 */
export const AcnStorageChangesLive: Layer.Layer<never, never, AcnChanges | ProjectStore | SessionInspector> =
  Layer.scopedDiscard(Effect.gen(function* () {
    const changes = yield* AcnChanges
    const projects = yield* ProjectStore
    const sessions = yield* SessionInspector
    const forward = (source: Stream.Stream<unknown>, queries: ReadonlyArray<string>) =>
      source.pipe(
        Stream.runForEach(() => Effect.forEach(queries, (query) => changes.publish({ query }), { discard: true })),
        Effect.forkScoped,
      )
    yield* forward(projects.changes, projectChangeQueries)
    yield* forward(sessions.changes, sessionChangeQueries)
  }))
