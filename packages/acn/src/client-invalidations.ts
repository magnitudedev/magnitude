import type { ClientInvalidation, MirroredStateInvalidation } from "@magnitudedev/acn-protocol"
import { Stream } from "effect"

/** Multiplexes independent connection-global invalidations onto one transport. */
export const mergeClientInvalidations = <MirrorError, ProjectError, SessionError>(sources: {
  readonly mirrors: Stream.Stream<MirroredStateInvalidation, MirrorError>
  readonly projects: Stream.Stream<unknown, ProjectError>
  readonly sessions: Stream.Stream<unknown, SessionError>
}): Stream.Stream<ClientInvalidation, MirrorError | ProjectError | SessionError> => {
  type InvalidationError = MirrorError | ProjectError | SessionError
  return Stream.mergeAll([
    sources.mirrors.pipe(
      Stream.map((invalidation): ClientInvalidation => ({
        _tag: "MirroredState",
        invalidation,
      })),
      Stream.mapError((error): InvalidationError => error),
    ),
    sources.projects.pipe(
      Stream.as<ClientInvalidation>({ _tag: "Projects" }),
      Stream.mapError((error): InvalidationError => error),
    ),
    sources.sessions.pipe(
      Stream.as<ClientInvalidation>({ _tag: "Sessions" }),
      Stream.mapError((error): InvalidationError => error),
    ),
  ], { concurrency: 3 })
}
