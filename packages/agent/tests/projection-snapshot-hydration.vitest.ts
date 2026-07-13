import { describe, expect, it } from 'vitest'
import { Effect } from 'effect'
import type { EventCursor, ProjectionSnapshotRestorePlan, Timestamped } from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import type { ChatPersistenceService, SessionMetadata } from '../src/persistence/chat-persistence-service'
import { loadProjectionSnapshotHydrationEvents } from '../src/runtime/projection-snapshot-hydration'

const event = (
  timestamp: number
): Timestamped<AppEvent> => ({
  type: 'wake',
  forkId: null,
  timestamp
})

const metadata: SessionMetadata = {
  sessionId: 'session-1',
  chatName: 'Test',
  workingDirectory: '/tmp',
  gitBranch: null,
  created: '2026-01-01T00:00:00.000Z',
  updated: '2026-01-01T00:00:00.000Z',
  initialVersion: '0.0.1',
  lastActiveVersion: '0.0.1',
}

const makePersistence = (options: {
  readonly events: readonly Timestamped<AppEvent>[]
  readonly projectionSnapshot: unknown | null
}): ChatPersistenceService => ({
  loadEvents: () => Effect.succeed([...options.events]),
  loadEventsAfterCursor: (cursor) =>
    Effect.succeed(
      options.events[cursor.index]?.timestamp === cursor.timestamp
        ? [...options.events.slice(cursor.index + 1)]
        : null
    ),
  persistNewEvents: () => Effect.succeed(null),
  loadProjectionSnapshot: () => Effect.succeed(options.projectionSnapshot),
  saveProjectionSnapshot: () => Effect.void,
  getSessionMetadata: () => Effect.succeed(metadata),
  saveSessionMetadata: () => Effect.void,
})

const makeEngine = (options: {
  readonly restorePlan: ProjectionSnapshotRestorePlan
  readonly onPrepare?: (snapshot: unknown) => void
}): {
  readonly prepareProjectionSnapshotRestore: (snapshot: unknown) => Effect.Effect<ProjectionSnapshotRestorePlan>
} => ({
  prepareProjectionSnapshotRestore: (snapshot) =>
    Effect.sync(() => {
      options.onPrepare?.(snapshot)
      return options.restorePlan
    })
})

describe('projection snapshot hydration', () => {
  it('restores a projection snapshot when its cursor is at the event-log tip', async () => {
    const events = [event(100)]
    const cursor: EventCursor = { index: 0, timestamp: 100 }
    let committed = false
    let markedHydrating = false

    const result = await Effect.runPromise(
      loadProjectionSnapshotHydrationEvents({
        persistence: makePersistence({
          events,
          projectionSnapshot: { eventCursor: cursor }
        }),
        engine: makeEngine({
          restorePlan: {
            eventCursor: cursor,
            commit: Effect.sync(() => {
              committed = true
            })
          }
        }),
        beforeSnapshotCommit: Effect.sync(() => {
          markedHydrating = true
        })
      })
    )

    expect(result).toEqual({
      events: [],
      restoredFromSnapshot: true
    })
    expect(markedHydrating).toBe(true)
    expect(committed).toBe(true)
  })

  it('replays the full event log instead of restoring a snapshot with an event suffix', async () => {
    const events = [event(100), event(200)]
    const cursor: EventCursor = { index: 0, timestamp: 100 }
    const warnings: string[] = []
    let committed = false

    const result = await Effect.runPromise(
      loadProjectionSnapshotHydrationEvents({
        persistence: makePersistence({
          events,
          projectionSnapshot: { eventCursor: cursor }
        }),
        engine: makeEngine({
          restorePlan: {
            eventCursor: cursor,
            commit: Effect.sync(() => {
              committed = true
            })
          }
        }),
        warn: ({ message }) => {
          warnings.push(message)
        }
      })
    )

    expect(result).toEqual({
      events,
      restoredFromSnapshot: false
    })
    expect(committed).toBe(false)
    expect(warnings).toContain('Projection snapshot is behind the event log; replaying full log to keep addressed state consistent')
  })
})
