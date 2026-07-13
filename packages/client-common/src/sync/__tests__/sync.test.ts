import { describe, it, expect } from 'vitest'
import type { DisplayMessage, DisplayState as SdkDisplayState, DisplayViewShape } from '@magnitudedev/sdk'
import { shareRefs } from '../share-refs'
import { ReferencePreservingStore } from '../store'
import { createDisplayViewStore } from '../display-view-store'

type KeyedDisplayMessage = DisplayMessage & { readonly _key: string }

function hasInjectedKey(message: DisplayMessage): message is KeyedDisplayMessage {
  return '_key' in message && typeof message._key === 'string'
}

function expectMessage(message: DisplayMessage | undefined): DisplayMessage {
  if (message === undefined) {
    throw new Error('Expected display message')
  }
  return message
}

describe('shareRefs', () => {
  describe('null old values', () => {
    it('handles shareRefs(null, []) without crashing', () => {
      const result = shareRefs<readonly unknown[] | null>(null, [])
      expect(result).toEqual([])
    })

    it('handles shareRefs(null, [{ _key: "a" }]) without crashing', () => {
      const result = shareRefs<readonly { readonly _key: string; readonly content: string }[] | null>(
        null,
        [{ _key: 'a', content: 'hi' }],
      )
      expect(result).toEqual([{ _key: 'a', content: 'hi' }])
    })

    it('handles null old field with array new value in object', () => {
      type State = { readonly items: readonly { readonly _key: string; readonly v: number }[] | null }
      const old: State = { items: null }
      const next: State = { items: [{ _key: 'x', v: 1 }] }
      const result = shareRefs(old, next)
      expect(result.items).toEqual([{ _key: 'x', v: 1 }])
    })
  })

  describe('scalars', () => {
    it('preserves scalar references by value equality', () => {
      expect(shareRefs('hello', 'hello')).toBe('hello')
      expect(shareRefs('hello', 'world')).toBe('world')
      expect(shareRefs(42, 42)).toBe(42)
      expect(shareRefs(42, 43)).toBe(43)
      expect(shareRefs(true, true)).toBe(true)
      expect(shareRefs(true, false)).toBe(false)
      expect(shareRefs(null, null)).toBe(null)
      expect(shareRefs<null>(undefined, null)).toBe(null)
    })
  })

  describe('plain objects', () => {
    it('returns old ref when all fields are equal', () => {
      const old = { a: 1, b: 'hello', c: { d: true } }
      const next = { a: 1, b: 'hello', c: { d: true } }
      const result = shareRefs(old, next)
      expect(result).toBe(old)
    })

    it('returns new object with shared refs for unchanged fields when one field changes', () => {
      const old = { a: 1, b: 'hello', nested: { x: 1 } }
      const next = { a: 2, b: 'hello', nested: { x: 1 } }
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result).toEqual(next)
      expect(result.b).toBe(old.b) // same string ref (trivial for primitives)
      expect(result.nested).toBe(old.nested) // same object ref — structural sharing!
    })

    it('shares nested unchanged objects recursively', () => {
      const old = { a: { b: { c: { d: 1 } } } }
      const next = { a: { b: { c: { d: 1 } } } }
      const result = shareRefs(old, next)
      expect(result).toBe(old) // all equal → same top-level ref
    })

    it('shares deeply nested unchanged sibling objects', () => {
      const old = { left: { value: 1 }, right: { value: 2 } }
      const next = { left: { value: 1 }, right: { value: 3 } }
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result.left).toBe(old.left) // unchanged — same ref
      expect(result.right).not.toBe(old.right) // changed — new ref
    })
  })

  describe('arrays with _key elements', () => {
    it('preserves refs for unchanged items when one item changes', () => {
      const old = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
      ]
      const next = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world!' },
      ]
      const result = shareRefs(old, next)
      expect(result).not.toBe(old) // array changed
      expect(result[0]).toBe(old[0]) // item a unchanged — same ref
      expect(result[1]).not.toBe(old[1]) // item b changed — new ref
    })

    it('preserves refs when appending a new item', () => {
      const old = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
      ]
      const next = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
        { _key: 'c', content: 'new' },
      ]
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result[0]).toBe(old[0])
      expect(result[1]).toBe(old[1])
      expect(result[2]).toBe(next[2])
    })

    it('preserves refs for remaining items when one is removed', () => {
      const old = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
        { _key: 'c', content: '!' },
      ]
      const next = [
        { _key: 'a', content: 'hello' },
        { _key: 'c', content: '!' },
      ]
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result[0]).toBe(old[0])
      expect(result[1]).toBe(old[2]) // matched by _key, not index
    })

    it('preserves item refs when order changes', () => {
      const old = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
      ]
      const next = [
        { _key: 'b', content: 'world' },
        { _key: 'a', content: 'hello' },
      ]
      const result = shareRefs(old, next)
      expect(result).not.toBe(old) // order changed → new array
      expect(result[0]).toBe(old[1]) // b is same object
      expect(result[1]).toBe(old[0]) // a is same object
    })

    it('returns same array ref when nothing changes', () => {
      const old = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
      ]
      const next = [
        { _key: 'a', content: 'hello' },
        { _key: 'b', content: 'world' },
      ]
      const result = shareRefs(old, next)
      expect(result).toBe(old)
    })
  })

  describe('arrays without _key elements', () => {
    it('compares by index', () => {
      const old = [1, 2, 3]
      const next = [1, 2, 4]
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result[0]).toBe(old[0])
      expect(result[1]).toBe(old[1])
      expect(result[2]).toBe(next[2])
    })

    it('returns same array ref when all elements equal', () => {
      const old = [1, 2, 3]
      const next = [1, 2, 3]
      expect(shareRefs(old, next)).toBe(old)
    })
  })

  describe('complex DisplayState-like objects', () => {
    interface Msg { readonly _key: string; readonly type: string; readonly content: string; readonly timestamp: number }
    interface DisplayState {
      readonly status: 'idle' | 'streaming'
      readonly messages: readonly Msg[]
      readonly streamingMessageId: string | null
      readonly showButton: 'send' | 'stop'
      readonly statusBar: { readonly active: boolean; readonly activity: string | null }
    }

    it('preserves messages array ref when only statusBar changes', () => {
      const old: DisplayState = {
        status: 'idle',
        messages: [{ _key: 'm1', type: 'user', content: 'hi', timestamp: 1 }],
        streamingMessageId: null,
        showButton: 'send',
        statusBar: { active: false, activity: null },
      }
      const next: DisplayState = {
        status: 'streaming',
        messages: [{ _key: 'm1', type: 'user', content: 'hi', timestamp: 1 }],
        streamingMessageId: 'm2',
        showButton: 'stop',
        statusBar: { active: true, activity: 'Thinking...' },
      }
      const result = shareRefs(old, next)
      expect(result).not.toBe(old)
      expect(result.messages).toBe(old.messages) // unchanged — same array ref!
      expect(result.messages[0]).toBe(old.messages[0]) // message unchanged
      expect(result.statusBar).not.toBe(old.statusBar) // changed
    })

    it('preserves message refs when only one message content changes', () => {
      const old: DisplayState = {
        status: 'streaming',
        messages: [
          { _key: 'm1', type: 'user', content: 'hi', timestamp: 1 },
          { _key: 'm2', type: 'assistant', content: 'Hello', timestamp: 2 },
          { _key: 'm3', type: 'assistant', content: 'Hello w', timestamp: 3 },
        ],
        streamingMessageId: 'm3',
        showButton: 'stop',
        statusBar: { active: true, activity: 'Thinking...' },
      }
      const next: DisplayState = {
        ...old,
        messages: [
          { _key: 'm1', type: 'user', content: 'hi', timestamp: 1 },
          { _key: 'm2', type: 'assistant', content: 'Hello', timestamp: 2 },
          { _key: 'm3', type: 'assistant', content: 'Hello world!', timestamp: 3 },
        ],
      }
      const result = shareRefs(old, next)
      expect(result.messages).not.toBe(old.messages) // array changed
      expect(result.messages[0]).toBe(old.messages[0]) // m1 same ref
      expect(result.messages[1]).toBe(old.messages[1]) // m2 same ref
      expect(result.messages[2]).not.toBe(old.messages[2]) // m3 changed — new ref
      expect(result.statusBar).toBe(old.statusBar) // unchanged
    })
  })
})

describe('ReferencePreservingStore', () => {
  it('notifies subscribers on state change', () => {
    const store = new ReferencePreservingStore({ value: 0 })
    let notifications = 0
    store.subscribe(() => notifications++)

    store.set({ value: 1 })
    expect(notifications).toBe(1)
    expect(store.get().value).toBe(1)
  })

  it('does not notify when state is identical (no-op)', () => {
    const store = new ReferencePreservingStore({ value: 0, list: [{ id: 'a', content: 'x' }] })
    let notifications = 0
    store.subscribe(() => notifications++)

    // Push identical state
    store.set({ value: 0, list: [{ id: 'a', content: 'x' }] })
    expect(notifications).toBe(0) // shareRefs returns same top-level ref
  })

  it('preserves references through set()', () => {
    const initial = {
      messages: [{ id: 'm1', content: 'hello' }],
      status: 'idle' as 'idle' | 'streaming',
    }
    const store = new ReferencePreservingStore(initial)
    const m1Before = store.get().messages[0]

    // Update with same m1, different status
    store.set({
      messages: [{ id: 'm1', content: 'hello' }],
      status: 'streaming' as 'idle' | 'streaming',
    })

    const m1After = store.get().messages[0]
    expect(m1After).toBe(m1Before) // same object reference!
  })

  it('unsubscribe stops notifications', () => {
    const store = new ReferencePreservingStore({ value: 0 })
    let notifications = 0
    const unsub = store.subscribe(() => notifications++)

    store.set({ value: 1 })
    expect(notifications).toBe(1)

    unsub()
    store.set({ value: 2 })
    expect(notifications).toBe(1) // no new notification
  })

  it('update() applies a function to current state', () => {
    const store = new ReferencePreservingStore({ count: 0 })
    store.update(prev => ({ count: prev.count + 1 }))
    expect(store.get().count).toBe(1)
    store.update(prev => ({ count: prev.count + 1 }))
    expect(store.get().count).toBe(2)
  })
})

describe('createDisplayViewStore (with _key injection)', () => {
  const userMessage = (id: string, content: string, timestamp: number): DisplayMessage => ({
    id,
    type: 'user_message',
    content,
    timestamp,
    taskMode: false,
    attachments: [],
  })

  const assistantMessage = (id: string, content: string, timestamp: number): DisplayMessage => ({
    id,
    type: 'assistant_message',
    content,
    timestamp,
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

  it('injects _key from id on objects in arrays', () => {
    const store = createDisplayViewStore(makeDisplayState([
      userMessage('m1', 'hi', 1),
    ]), makeShape())

    const state = store.getSnapshot().state
    const message = expectMessage(state.timelines.root.messages.byId['m1'])
    expect(hasInjectedKey(message)).toBe(true)
    if (!hasInjectedKey(message)) return
    expect(message._key).toBe('m1')
  })

  it('preserves message ref on second set when message unchanged', () => {
    const initial = makeDisplayState([
      userMessage('m1', 'hi', 1),
    ])
    const store = createDisplayViewStore(initial, makeShape())
    const m1First = expectMessage(store.getSnapshot().state.timelines.root.messages.byId['m1'])

    // Same message ref, different status
    store.accept({
      shape: makeShape(),
      state: {
        ...initial,
        timelines: {
          root: {
            ...initial.timelines.root,
            mode: 'streaming',
            messages: { byId: { m1: m1First }, order: ['m1'] }, // reuse same ref (already has _key)
          },
        },
      },
    })

    const m1Second = expectMessage(store.getSnapshot().state.timelines.root.messages.byId['m1'])
    expect(m1Second).toBe(m1First) // same ref — _key already present, no spread
  })

  it('matches by _key when a message is removed from the middle', () => {
    const initial = makeDisplayState([
      userMessage('m1', 'first', 1),
      assistantMessage('m2', 'middle', 2),
      assistantMessage('m3', 'last', 3),
    ])
    const store = createDisplayViewStore(initial, makeShape())
    const m1Before = expectMessage(store.getSnapshot().state.timelines.root.messages.byId['m1'])
    const m3Before = expectMessage(store.getSnapshot().state.timelines.root.messages.byId['m3'])

    // Remove m2 — new objects (no _key yet, will be injected)
    store.accept({
      shape: makeShape(),
      state: {
        ...initial,
        timelines: {
          root: {
            ...initial.timelines.root,
            messages: normalizeMessages([
              userMessage('m1', 'first', 1),
              assistantMessage('m3', 'last', 3),
            ]),
          },
        },
      },
    })

    const msgs = store.getSnapshot().state.timelines.root.messages
    const first = expectMessage(msgs.byId['m1'])
    const second = expectMessage(msgs.byId['m3'])
    // m1 and m3 should be preserved by keyed-object match
    expect(hasInjectedKey(first)).toBe(true)
    expect(hasInjectedKey(second)).toBe(true)
    if (!hasInjectedKey(first) || !hasInjectedKey(second)) return
    expect(first._key).toBe('m1')
    expect(second._key).toBe('m3')
    expect(first).toBe(m1Before) // same ref — matched by _key
    expect(second).toBe(m3Before) // same ref — matched by _key
  })
})
