import {
  projectChangeQueries,
  sessionChangeQueries,
  type Change,
  type MirroredStateInvalidation,
} from "@magnitudedev/acn-protocol"
import { Stream } from "effect"

/**
 * Multiplexes every ACN change source onto one stream of pokes in the
 * clients' query-identity space. A mirror commit names its own query (the
 * mirror id is the query's tag) with its revision; a store commit names every
 * query it backs.
 */
export const mergeChanges = <MirrorError, ProjectError, SessionError>(sources: {
  readonly mirrors: Stream.Stream<MirroredStateInvalidation, MirrorError>
  readonly projects: Stream.Stream<unknown, ProjectError>
  readonly sessions: Stream.Stream<unknown, SessionError>
}): Stream.Stream<Change, MirrorError | ProjectError | SessionError> => {
  type ChangeError = MirrorError | ProjectError | SessionError
  return Stream.mergeAll([
    sources.mirrors.pipe(
      Stream.map((invalidation): Change => ({ query: invalidation.id, revision: invalidation.revision })),
      Stream.mapError((error): ChangeError => error),
    ),
    sources.projects.pipe(
      Stream.mapConcat((): ReadonlyArray<Change> => projectChangeQueries.map((query) => ({ query }))),
      Stream.mapError((error): ChangeError => error),
    ),
    sources.sessions.pipe(
      Stream.mapConcat((): ReadonlyArray<Change> => sessionChangeQueries.map((query) => ({ query }))),
      Stream.mapError((error): ChangeError => error),
    ),
  ], { concurrency: 3 })
}
