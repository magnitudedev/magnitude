import { Context, Data, Effect, type ParseResult } from 'effect'
import type { EventCursor } from '../core/event-cursor'
import type { AddressedError } from '../addressed/errors'
import type { ProjectionSnapshotEnvelope } from './make'

export class ProjectionSnapshotEnvelopeInvalid extends Data.TaggedError('ProjectionSnapshotEnvelopeInvalid')<{
  readonly cause: ParseResult.ParseError
}> {}

export class ProjectionSnapshotProjectionSetMismatch extends Data.TaggedError('ProjectionSnapshotProjectionSetMismatch')<{
  readonly missing: readonly string[]
  readonly extra: readonly string[]
}> {}

export class ProjectionSnapshotProjectionInvalid extends Data.TaggedError('ProjectionSnapshotProjectionInvalid')<{
  readonly projectionName: string
  readonly cause: ParseResult.ParseError
}> {}

export type ProjectionSnapshotInvalid =
  | ProjectionSnapshotEnvelopeInvalid
  | ProjectionSnapshotProjectionSetMismatch
  | ProjectionSnapshotProjectionInvalid

export interface ProjectionSnapshotRestorePlan {
  readonly eventCursor: EventCursor
  readonly commit: Effect.Effect<void>
}

export interface ProjectionSnapshotService {
  readonly captureProjectionSnapshot: (
    cursor: EventCursor,
    sessionId: string
  ) => Effect.Effect<ProjectionSnapshotEnvelope, ParseResult.ParseError | AddressedError>
  readonly prepareProjectionSnapshotRestore: (
    snapshot: unknown
  ) => Effect.Effect<ProjectionSnapshotRestorePlan, ProjectionSnapshotInvalid>
}

export const ProjectionSnapshotServiceTag = Context.GenericTag<ProjectionSnapshotService>(
  'ProjectionSnapshotService'
)
