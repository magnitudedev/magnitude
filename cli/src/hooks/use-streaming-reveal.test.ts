import { beforeEach, describe, expect, test, vi } from 'vitest'

/**
 * Drives the hook as a state machine with mocked React primitives.
 * The hook only uses useRef and useSyncExternalStore; mocking them lets us
 * assert the subscription choice (live tick vs noop) per lifecycle phase,
 * which a server render cannot exercise.
 */

const refs: { current: unknown }[] = []
let refIdx = 0
let chosenSubscribe: unknown = null

vi.mock('react', async () => {
  const actual = await vi.importActual<typeof import('react')>('react')
  return {
    ...actual,
    useRef: (initial: unknown) => {
      const i = refIdx++
      if (refs[i] === undefined) refs[i] = { current: initial }
      return refs[i]
    },
    useSyncExternalStore: (subscribe: unknown, getSnapshot: () => unknown) => {
      chosenSubscribe = subscribe
      return getSnapshot()
    },
  }
})

let tick = 0

vi.mock('@magnitudedev/client-common', async () => {
  const actual = await vi.importActual<typeof import('@magnitudedev/client-common')>('@magnitudedev/client-common')
  return {
    ...actual,
    subscribeAnimationTick: () => () => {},
    getAnimationTickSnapshot: () => tick,
  }
})

import { subscribeAnimationTick, subscribeAnimationNoop } from '@magnitudedev/client-common'
import { useStreamingReveal } from './use-streaming-reveal'

function render(
  content: string,
  isStreaming: boolean,
  isInterrupted?: boolean,
  initialDisplayedLength?: number,
) {
  refIdx = 0
  return useStreamingReveal(content, isStreaming, isInterrupted, initialDisplayedLength)
}

const isLive = () => chosenSubscribe === subscribeAnimationTick
const isNoop = () => chosenSubscribe === subscribeAnimationNoop

beforeEach(() => {
  refs.length = 0
  refIdx = 0
  chosenSubscribe = null
  tick = 100
})

describe('useStreamingReveal', () => {
  test('starts from provided initialDisplayedLength when mounting during active streaming', () => {
    const state = render('abcdefghij', true, undefined, 7)
    expect(state.displayedContent).toBe('abcdefg')
    expect(state.isCatchingUp).toBe(true)
    expect(isLive()).toBe(true)
  })

  test('defaults to empty reveal when mounting during active streaming without initialDisplayedLength', () => {
    const state = render('abcdefghij', true)
    expect(state.displayedContent).toBe('')
    expect(state.isCatchingUp).toBe(true)
    expect(isLive()).toBe(true)
  })

  test('mounts completed content fully revealed and does not subscribe to ticks', () => {
    const state = render('hello world, completed message', false)
    expect(state.displayedContent).toBe('hello world, completed message')
    expect(state.isCatchingUp).toBe(false)
    expect(state.showCursor).toBe(false)
    expect(isNoop()).toBe(true)
  })

  test('reveals during streaming, drains after stream end, then unsubscribes', () => {
    const content = 'x'.repeat(40)

    // Stream starts empty, content arrives
    render('', true)
    expect(isLive()).toBe(true)

    // Ticks advance the reveal while streaming
    tick++
    let state = render(content, true)
    expect(state.displayedContent.length).toBeGreaterThan(0)
    expect(state.displayedContent.length).toBeLessThan(content.length)
    expect(isLive()).toBe(true)

    // Stream ends mid-reveal — linear drain keeps the subscription
    tick++
    state = render(content, false)
    expect(isLive()).toBe(true)

    // Drain to completion
    for (let i = 0; i < 20 && state.isCatchingUp; i++) {
      tick++
      state = render(content, false)
    }
    expect(state.displayedContent).toBe(content)
    expect(state.isCatchingUp).toBe(false)

    // Caught up and idle — next render unsubscribes
    state = render(content, false)
    expect(isNoop()).toBe(true)
    expect(state.showCursor).toBe(false)
  })

  test('drains content growth on a component that never streamed', () => {
    const initial = 'short'
    let state = render(initial, false)
    expect(state.displayedContent).toBe(initial)
    expect(isNoop()).toBe(true)

    // Content grows without a streaming phase — must reveal and re-quiesce
    const grown = initial + '!'.repeat(30)
    state = render(grown, false)
    expect(isLive()).toBe(true)
    for (let i = 0; i < 20 && state.isCatchingUp; i++) {
      tick++
      state = render(grown, false)
    }
    expect(state.displayedContent).toBe(grown)
    state = render(grown, false)
    expect(isNoop()).toBe(true)
  })

  test('interrupt snaps to full content without subscribing', () => {
    render('', true)
    tick++
    const state = render('interrupted content', true, true)
    expect(state.displayedContent).toBe('interrupted content')
    expect(isNoop()).toBe(true)
  })
})
