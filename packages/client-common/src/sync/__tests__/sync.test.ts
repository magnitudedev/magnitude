import { describe, it, expect } from 'vitest'
import type { DisplayMessage, DisplayState as SdkDisplayState, DisplayViewShape } from '@magnitudedev/sdk'
import { createDisplayViewStore } from '../display-view-store'

describe('createDisplayViewStore', () => {
  const userMessage = (id: string, content: string, timestamp: number): DisplayMessage => ({
    id,
    type: 'user_message',
    content,
    timestamp,
    taskMode: false,
    attachments: [],
  })

  const normalizeMessages = (messages: readonly DisplayMessage[]) => ({
    byId: Object.fromEntries(messages.map((m) => [m.id, m])),
    order: messages.map((m) => m.id),
  })

  const emptyWindow = (count = 0) => ({
    start: 0,
    end: count,
    totalCount: count,
    hasMoreBefore: false,
    hasMoreAfter: false,
  })

  const emptyPresentation = () => ({
    mode: 'default' as const,
    entries: [],
    statusSlot: { kind: 'none' as const },
  })

  const makeDisplayState = (
    messages: readonly DisplayMessage[],
    mode: 'idle' | 'streaming' = 'idle',
  ): SdkDisplayState => ({
    session: { sessionId: 's1', title: 'Test', cwd: '/tmp' },
    timelines: {
      root: {
        mode,
        messages: normalizeMessages(messages),
        streamingMessageId: null,
        window: emptyWindow(messages.length),
        presentation: emptyPresentation(),
      },
    },
    actors: {},
    agents: {},
    tasks: { byId: {}, order: [], summary: { totalCount: 0, completedCount: 0, incompleteCount: 0 } },
  })

  const makeShape = (): DisplayViewShape => ({
    timelines: {
      root: { kind: 'tail', limit: 300, live: true, presentation: 'default' },
    },
  })

  it('returns initial snapshot', () => {
    const store = createDisplayViewStore(makeDisplayState([
      userMessage('m1', 'hi', 1),
    ]), makeShape())

    const state = store.getSnapshot().state
    expect(state.timelines.root.messages.byId['m1']).toMatchObject({
      type: 'user_message',
      content: 'hi',
    })
  })

  it('notifies subscribers on accept', () => {
    const initial = makeDisplayState([userMessage('m1', 'hi', 1)])
    const store = createDisplayViewStore(initial, makeShape())
    let notifications = 0
    store.subscribe(() => notifications++)

    store.accept({ shape: makeShape(), state: { ...initial, session: { sessionId: 's1', title: 'Changed', cwd: '/tmp' } } })
    expect(notifications).toBe(1)
    expect(store.getSnapshot().state.session.title).toBe('Changed')
  })

  it('does not notify when same ref is accepted', () => {
    const initial = makeDisplayState([userMessage('m1', 'hi', 1)])
    const store = createDisplayViewStore(initial, makeShape())
    let notifications = 0
    store.subscribe(() => notifications++)

    store.accept(store.getSnapshot())
    expect(notifications).toBe(0)
  })

  it('unsubscribe stops notifications', () => {
    const initial = makeDisplayState([userMessage('m1', 'hi', 1)])
    const store = createDisplayViewStore(initial, makeShape())
    let notifications = 0
    const unsub = store.subscribe(() => notifications++)

    store.accept({ shape: makeShape(), state: { ...initial, session: { sessionId: 's1', title: 'A', cwd: '/tmp' } } })
    expect(notifications).toBe(1)

    unsub()
    store.accept({ shape: makeShape(), state: { ...initial, session: { sessionId: 's1', title: 'B', cwd: '/tmp' } } })
    expect(notifications).toBe(1)
  })

  it('speculative mutate and remove', () => {
    const store = createDisplayViewStore(makeDisplayState([
      userMessage('m1', 'hi', 1),
    ]), makeShape())

    const handle = store.mutate(
      { owner: 'test', label: 'append' },
      (draft) => {
        draft.state.timelines.root.messages.byId['m2'] = userMessage('m2', 'world', 2)
        draft.state.timelines.root.messages.order = [
          ...draft.state.timelines.root.messages.order,
          'm2',
        ]
      },
    )

    // Should see the speculative message
    expect(store.getSnapshot().state.timelines.root.messages.byId['m2']).toMatchObject({
      type: 'user_message',
      content: 'world',
    })

    handle.remove()

    // Should be gone after remove
    expect(store.getSnapshot().state.timelines.root.messages.byId['m2']).toBeUndefined()
  })

  it('clear removes all speculative transactions', () => {
    const store = createDisplayViewStore(makeDisplayState([
      userMessage('m1', 'hi', 1),
    ]), makeShape())

    store.mutate(
      { owner: 'test' },
      (draft) => {
        draft.state.timelines.root.messages.byId['m2'] = userMessage('m2', 'world', 2)
        draft.state.timelines.root.messages.order = [
          ...draft.state.timelines.root.messages.order,
          'm2',
        ]
      },
    )

    expect(store.getSnapshot().state.timelines.root.messages.byId['m2']).toBeDefined()

    store.clear()

    expect(store.getSnapshot().state.timelines.root.messages.byId['m2']).toBeUndefined()
  })
})
