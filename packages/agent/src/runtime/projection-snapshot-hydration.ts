import { Effect } from 'effect'
import type { ProjectionSnapshotRestorePlan, Timestamped } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import type { ChatPersistenceService, PersistenceError } from '../persistence/chat-persistence-service'

export interface ProjectionSnapshotHydrationEngine {
  readonly prepareProjectionSnapshotRestore: (
    snapshot: unknown
  ) => Effect.Effect<ProjectionSnapshotRestorePlan, unknown>
}

export interface ProjectionSnapshotHydrationDecision {
  readonly events: Timestamped<AppEvent>[]
  readonly restoredFromSnapshot: boolean
}

export interface ProjectionSnapshotHydrationWarning {
  readonly fields: Record<string, unknown>
  readonly message: string
}

export interface ProjectionSnapshotHydrationOptions {
  readonly persistence: ChatPersistenceService
  readonly engine: ProjectionSnapshotHydrationEngine
  readonly beforeSnapshotCommit?: Effect.Effect<void>
  readonly warn?: (warning: ProjectionSnapshotHydrationWarning) => void
}

const warn = (
  emit: ((warning: ProjectionSnapshotHydrationWarning) => void) | undefined,
  fields: Record<string, unknown>,
  message: string
): Effect.Effect<void> =>
  Effect.sync(() => emit?.({ fields, message }))

export const loadProjectionSnapshotHydrationEvents = (
  options: ProjectionSnapshotHydrationOptions
): Effect.Effect<ProjectionSnapshotHydrationDecision, PersistenceError> =>
  Effect.gen(function* () {
    const { persistence, engine } = options
    const snapshot = yield* persistence.loadProjectionSnapshot().pipe(
      Effect.catchAll((error) =>
        warn(options.warn, { error }, 'Failed to load projection snapshot; replaying events').pipe(
          Effect.as(null)
        )
      )
    )

    if (!snapshot) {
      return {
        events: yield* persistence.loadEvents(),
        restoredFromSnapshot: false
      }
    }

    const restorePlan = yield* engine.prepareProjectionSnapshotRestore(snapshot).pipe(
      Effect.catchAll((error) =>
        warn(options.warn, { error }, 'Invalid projection snapshot; replaying events').pipe(
          Effect.as(null)
        )
      )
    )

    if (!restorePlan) {
      return {
        events: yield* persistence.loadEvents(),
        restoredFromSnapshot: false
      }
    }

    const suffix = yield* persistence.loadEventsAfterCursor(restorePlan.eventCursor).pipe(
      Effect.catchAll((error) =>
        warn(options.warn, { error }, 'Failed to load event suffix after projection snapshot; replaying events').pipe(
          Effect.as(null)
        )
      )
    )

    if (suffix === null) {
      return {
        events: yield* persistence.loadEvents(),
        restoredFromSnapshot: false
      }
    }

    if (suffix.length > 0) {
      yield* warn(
        options.warn,
        {
          eventCursor: restorePlan.eventCursor,
          suffixEventCount: suffix.length,
        },
        'Projection snapshot is behind the event log; replaying full log to keep addressed state consistent'
      )
      return {
        events: yield* persistence.loadEvents(),
        restoredFromSnapshot: false
      }
    }

    if (options.beforeSnapshotCommit) {
      yield* options.beforeSnapshotCommit
    }
    yield* restorePlan.commit

    return {
      events: suffix,
      restoredFromSnapshot: true
    }
  })
