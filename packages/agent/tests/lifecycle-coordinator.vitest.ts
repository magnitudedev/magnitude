import { describe, expect, it } from 'vitest'
import { Effect, Layer, PubSub, Stream } from 'effect'
import {
  EventBusCoreTag,
  EventSinkTag,
  HydrationContext,
  ProjectionSnapshotServiceTag,
  type EventBusCoreService,
  type EventCursor,
  type EventSinkService,
  type Timestamped,
} from '@magnitudedev/event-core'

import type { AppEvent } from '../src/events'
import {
  ChatPersistence,
  type ChatPersistenceService,
  type SessionMetadata,
} from '../src/persistence/chat-persistence-service'
import { TurnProjection } from '../src/projections/turn'
import {
  LifecycleCoordinator,
  LifecycleCoordinatorTag,
} from '../src/workers/lifecycle-coordinator'

const makeEvent = (id: string, timestamp: number): Timestamped<AppEvent> => ({
  type: 'user_message',
  forkId: null,
  messageId: id,
  content: [{ _tag: 'TextPart', text: id }],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false,
  timestamp,
} as Timestamped<AppEvent>)

const metadata: SessionMetadata = {
  sessionId: 'session-1',
  chatName: 'Lifecycle Test',
  workingDirectory: '/tmp',
  gitBranch: null,
  created: '2026-01-01T00:00:00.000Z',
  updated: '2026-01-01T00:00:00.000Z',
  initialVersion: 'test',
  lastActiveVersion: 'test',
}

describe('LifecycleCoordinator', () => {
  it('persists projection snapshots through the turn-terminated lifecycle flush', async () => {
    const calls: string[] = []
    const cursor: EventCursor = { index: 0, timestamp: 100 }
    let pending = [makeEvent('msg-1', 100)]

    await Effect.runPromise(Effect.gen(function* () {
      const eventPubSub = yield* PubSub.unbounded<Timestamped<AppEvent>>()
      const turnTerminatedPubSub = yield* PubSub.unbounded<{
        forkId: string | null
        turnId: string
        reason: 'completed' | 'cancelled' | 'error'
        triggersQueued: boolean
      }>()

      const eventBus: Pick<EventBusCoreService<AppEvent>, 'checkpoint' | 'subscribeToTypes' | 'stream' | 'subscribe' | 'publish'> = {
        publish: () => Effect.void,
        checkpoint: (effect) =>
          Effect.gen(function* () {
            calls.push('checkpoint:start')
            const result = yield* effect
            calls.push('checkpoint:end')
            return result
          }),
        subscribeToTypes: (types) =>
          Stream.fromPubSub(eventPubSub).pipe(
            Stream.filter((event) => types.some((type) => type === event.type))
          ) as never,
        stream: Stream.fromPubSub(eventPubSub),
        subscribe: () => Effect.succeed(Stream.fromPubSub(eventPubSub)),
      }

      const eventSink: Pick<EventSinkService<AppEvent>, 'append' | 'readPending' | 'drainPending' | 'prependEvents'> = {
        append: (event) => Effect.sync(() => {
          pending.push(event)
        }),
        readPending: () => Effect.sync(() => [...pending]),
        drainPending: () => Effect.sync(() => {
          calls.push('drain')
          const drained = pending
          pending = []
          return drained
        }),
        prependEvents: (events) => Effect.sync(() => {
          calls.push(`requeue:${events.length}`)
          pending = [...events, ...pending]
        }),
      }

      const persistence: ChatPersistenceService = {
        loadEvents: () => Effect.succeed([]),
        loadEventsAfterCursor: () => Effect.succeed([]),
        persistNewEvents: (events) => Effect.sync(() => {
          calls.push(`persist:${events.length}`)
          return cursor
        }),
        loadProjectionSnapshot: () => Effect.succeed(null),
        saveProjectionSnapshot: (snapshot) => Effect.sync(() => {
          calls.push(`save:${(snapshot as { eventCursor: EventCursor }).eventCursor.index}`)
        }),
        getSessionMetadata: () => Effect.sync(() => {
          calls.push('metadata')
          return metadata
        }),
        saveSessionMetadata: () => Effect.void,
      }

      const engine = {
        captureProjectionSnapshot: (snapshotCursor: EventCursor, sessionId: string) =>
          Effect.sync(() => {
            calls.push(`snapshot:${sessionId}:${snapshotCursor.index}`)
            return {
              sessionId,
              engineName: 'CodingAgent',
              schemaVersion: 'test',
              eventCursor: snapshotCursor,
              projections: {},
            }
          }),
      }

      const deps = Layer.mergeAll(
        HydrationContext.Default,
        Layer.succeed(EventBusCoreTag<AppEvent>(), eventBus as EventBusCoreService<AppEvent>),
        Layer.succeed(EventSinkTag<AppEvent>(), eventSink as EventSinkService<AppEvent>),
        Layer.succeed(ChatPersistence, persistence),
        Layer.succeed(ProjectionSnapshotServiceTag, engine as never),
        Layer.succeed(TurnProjection.signals.turnTerminated.tag, turnTerminatedPubSub),
      )

      yield* Effect.scoped(
        Effect.gen(function* () {
          yield* LifecycleCoordinatorTag
          yield* Effect.sleep('10 millis')
          yield* PubSub.publish(turnTerminatedPubSub, {
            forkId: null,
            turnId: 'turn-1',
            reason: 'completed',
            triggersQueued: false,
          })
          yield* Effect.sleep('150 millis')
        }).pipe(
          Effect.provide(LifecycleCoordinator.Layer.pipe(Layer.provide(deps)))
        )
      )
    }))

    expect(calls.slice(0, 7)).toEqual([
      'checkpoint:start',
      'drain',
      'metadata',
      'persist:1',
      'snapshot:session-1:0',
      'save:0',
      'checkpoint:end',
    ])
  })
})
