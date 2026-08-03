import { describe, expect, it } from 'vitest'
import { Effect, Option, Schema } from 'effect'
import {
  DisplayStateSchema,
  DisplayTimelineSchema,
  StreamEventSchema,
  type DisplayMessage,
  type DisplayState,
  type DisplayTimeline,
  type DisplayViewShape,
  type StreamEvent,
} from '@magnitudedev/sdk'
import { applyStreamEvent } from '../apply-stream-event'
import { appendMessageToTimeline, createDisplayViewStore } from '../display-view-store'


const normalizedMessages = (messages: readonly DisplayMessage[]): DisplayTimeline['messages'] => ({
  byId: Object.fromEntries(messages.map((m) => [m.id, m])),
  order: messages.map((m) => m.id),
})

const emptyWindow = (count = 0): DisplayTimeline['window'] => ({
  start: 0,
  end: count,
  totalCount: count,
  hasMoreBefore: false,
  hasMoreAfter: false,
})

const emptyPresentation = (): DisplayTimeline['presentation'] => ({
  mode: 'default',
  entries: [],
  statusSlot: { kind: 'none' },
})

const timelineFromMessages = (
  messages: readonly DisplayMessage[],
  mode: DisplayTimeline['mode'] = 'idle',
): DisplayTimeline => ({
  mode,
  messages: normalizedMessages(messages),
  streamingMessageId: null,
  window: emptyWindow(messages.length),
  presentation: emptyPresentation(),
})

const baseState = (): DisplayState => ({
  session: { sessionId: 's1', title: null, cwd: '/tmp' },
  timelines: {
    root: timelineFromMessages([]),
  },
  actors: {},
  agents: {},
  tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
})

const baseShape = (): DisplayViewShape => ({
  timelines: {
    root: { kind: 'tail', limit: 300, live: true, presentation: 'default' },
  },
})

describe('applyStreamEvent', () => {
  it('round-trips decoded patch value payloads through the stream schema', () => {
    const event: StreamEvent = {
      _tag: 'patch',
      ops: [{ op: 'replace', path: ['state', 'session', 'title'], value: 'Review staged changes' }],
    }

    const encoded = Schema.encodeSync(StreamEventSchema)(event)

    expect(Schema.decodeUnknownSync(StreamEventSchema)(encoded)).toEqual(event)
  })

  it('round-trips full state events containing tool messages', () => {
    const state: DisplayState = {
      ...baseState(),
      timelines: {
        root: {
          ...baseState().timelines.root,
          messages: normalizedMessages([{
            id: 'tool-1',
            type: 'tool',
            toolKey: 'spawnWorker',
            cluster: Option.none(),
            presentation: Option.none(),
            filter: Option.none(),
            resultFilePath: Option.none(),
            timestamp: 1,
          }]),
        },
      },
    }
    const event: StreamEvent = { _tag: 'state', shape: baseShape(), state }

    const encoded = Schema.encodeSync(StreamEventSchema)(event)
    if (encoded._tag !== 'state') {
      throw new Error('Expected encoded state event')
    }
    expect(encoded.state.timelines.root?.messages.byId['tool-1']).toMatchObject({
      type: 'tool',
      toolKey: 'spawnWorker',
    })

    const decoded = Schema.decodeUnknownSync(StreamEventSchema)(encoded)
    if (decoded._tag !== 'state') {
      throw new Error('Expected decoded state event')
    }
    const message = decoded.state.timelines.root?.messages.byId['tool-1']
    if (message?.type !== 'tool') {
      throw new Error('Expected decoded tool message')
    }

    expect(message.toolKey).toBe('spawnWorker')
  })

  it('applies snapshot-rooted patches and decodes Option fields', () => {
    const initial: DisplayState = {
      ...baseState(),
      tasks: {
        byId: {
          root: {
            rowId: 'row-root',
            kind: 'task',
            taskId: 'root',
            title: 'Root',
            status: 'pending',
            parentId: Option.none(),
            depth: 0,
            updatedAt: 1,
            assignee: { kind: 'none' },
          },
        },
        order: ['root'],
        summary: { totalCount: 1, completedCount: 0, incompleteCount: 1 },
      },
    }
    const store = createDisplayViewStore(initial, baseShape())

    Effect.runSync(applyStreamEvent(
      store,
      { _tag: 'patch', ops: [{ op: 'replace', path: ['state', 'tasks', 'byId', 'root', 'parentId'], value: 'parent' }] },
      null,
      's1',
      'view-1',
    ))

    const root = store.getSnapshot().state.tasks.byId.root
    if (!root) {
      throw new Error('Expected patched task to exist')
    }

    const encodedRoot = Schema.encodeSync(DisplayStateSchema)(store.getSnapshot().state).tasks.byId.root
    if (!encodedRoot) {
      throw new Error('Expected encoded patched task to exist')
    }

    expect(Option.match(root.parentId, { onNone: () => 'none', onSome: (value) => value })).toBe('parent')
    expect(encodedRoot.parentId).toBe('parent')
    expect(store.getSnapshot().shape).toEqual(baseShape())
  })

  it('applies patches that add a worker timeline to an existing display view', () => {
    const store = createDisplayViewStore(baseState(), baseShape())
    const workerTimeline: DisplayTimeline = {
      mode: 'idle',
      messages: normalizedMessages([{
        id: 'worker-msg-1',
        type: 'assistant_message',
        content: 'worker loaded',
        timestamp: 10,
      }]),
      streamingMessageId: null,
      window: emptyWindow(1),
      presentation: emptyPresentation(),
    }

    Effect.runSync(applyStreamEvent(
      store,
      {
        _tag: 'patch',
        ops: [
          {
            op: 'replace',
            path: ['shape', 'timelines', 'worker-1'],
            value: { kind: 'tail', limit: 200, live: true, presentation: 'default' },
          },
          {
            op: 'replace',
            path: ['state', 'timelines', 'worker-1'],
            value: Schema.encodeSync(DisplayTimelineSchema)(workerTimeline),
          },
        ],
      },
      null,
      's1',
      'view-1',
    ))

    expect(store.getSnapshot().shape.timelines['worker-1']).toEqual({ kind: 'tail', limit: 200, live: true, presentation: 'default' })
    expect(store.getSnapshot().state.timelines['worker-1']).toEqual(workerTimeline)
  })

  it('applies patches through defaulted presentation fields', () => {
    const store = createDisplayViewStore(baseState(), baseShape())
    const entry = {
      kind: 'message' as const,
      id: 'entry-1',
      messageId: 'message-1',
      timestamp: 1,
      role: 'assistant' as const,
      streaming: true,
      interrupted: false,
      nextMessageInterrupted: false,
    }

    Effect.runSync(applyStreamEvent(
      store,
      {
        _tag: 'patch',
        ops: [{
          op: 'add',
          path: ['state', 'timelines', 'root', 'presentation', 'entries', 0],
          value: entry,
        }],
      },
      null,
      's1',
      'view-1',
    ))

    expect(store.getSnapshot().state.timelines.root?.presentation.entries).toEqual([entry])

    Effect.runSync(applyStreamEvent(
      store,
      {
        _tag: 'patch',
        ops: [{
          op: 'replace',
          path: ['state', 'timelines', 'root', 'presentation', 'entries', 0, 'streaming'],
          value: false,
        }],
      },
      null,
      's1',
      'view-1',
    ))

    expect(store.getSnapshot().state.timelines.root?.presentation.entries[0]).toEqual({
      ...entry,
      streaming: false,
    })
  })

  it('forwards queued-message restore events to the caller', () => {
    const store = createDisplayViewStore(baseState(), baseShape())
    const restored: unknown[] = []

    Effect.runSync(applyStreamEvent(
      store,
      {
        _tag: 'restore_queued_messages',
        forkId: null,
        messages: [{ id: 'queued-1', content: 'restore me', taskMode: false }],
      },
      null,
      's1',
      'view-1',
      (payload) => restored.push(payload),
    ))

    expect(restored).toEqual([{
      forkId: null,
      messages: [{ id: 'queued-1', content: 'restore me', taskMode: false }],
    }])
    expect(store.getSnapshot().state).toEqual(baseState())
  })

  it('replays mergeable speculative messages over unrelated accepted sync', () => {
    const store = createDisplayViewStore(baseState(), baseShape())
    const m2: DisplayMessage = {
      id: 'm2',
      type: 'user_message',
      content: 'speculative',
      timestamp: 2,
      taskMode: false,
      attachments: [],
    }

    store.mutate({ owner: 'test' }, (draft) => {
      draft.state.timelines.root = appendMessageToTimeline(draft.state.timelines.root!, m2)
    })

    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(['m2'])

    const m1: DisplayMessage = {
      id: 'm1',
      type: 'user_message',
      content: 'accepted',
      timestamp: 1,
      taskMode: false,
      attachments: [],
    }
    store.accept({
      shape: baseShape(),
      state: {
        ...baseState(),
        timelines: {
          root: appendMessageToTimeline(baseState().timelines.root!, m1),
        },
      },
    })

    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(['m1', 'm2'])
  })

  it('retires a speculative message when accepted sync writes the same message id', () => {
    const store = createDisplayViewStore(baseState(), baseShape())
    const m2: DisplayMessage = {
      id: 'm2',
      type: 'user_message',
      content: 'speculative',
      timestamp: 2,
      taskMode: false,
      attachments: [],
    }

    store.mutate({ owner: 'test' }, (draft) => {
      draft.state.timelines.root = appendMessageToTimeline(draft.state.timelines.root!, m2)
    })

    store.accept({
      shape: baseShape(),
      state: {
        ...baseState(),
        timelines: {
          root: appendMessageToTimeline(baseState().timelines.root!, {
            ...m2,
            content: 'accepted',
          }),
        },
      },
    })

    const renderedMessage = store.getSnapshot().state.timelines.root?.messages.byId.m2
    if (renderedMessage?.type !== 'user_message') {
      throw new Error('Expected rendered user message')
    }
    expect(renderedMessage.content).toBe('accepted')
    expect(store.getSnapshot().state.timelines.root?.messages.order).toEqual(['m2'])
  })
})
