import { beforeEach, describe, expect, test, vi } from 'vitest'

/**
 * Drives the hook as a state machine with mocked React primitives.
 * The hook only uses useRef and useSyncExternalStore; mocking them lets us
 * assert the subscription choice (live clock vs noop) per lifecycle phase,
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

let animationTime = 0

vi.mock('@magnitudedev/client-common', async () => {
  const actual = await vi.importActual<typeof import('@magnitudedev/client-common')>('@magnitudedev/client-common')
  return {
    ...actual,
    subscribeAnimationClock: () => () => {},
    getAnimationTimeSnapshot: () => animationTime,
  }
})

import { subscribeAnimationClock, subscribeAnimationNoop } from '@magnitudedev/client-common'
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

const isLive = () => chosenSubscribe === subscribeAnimationClock
const isNoop = () => chosenSubscribe === subscribeAnimationNoop

beforeEach(() => {
  refs.length = 0
  refIdx = 0
  chosenSubscribe = null
  animationTime = 8_000
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

  test('catches up proportionally on every animation frame, then snaps on completion', () => {
    const content = 'x'.repeat(100)

    // Stream starts empty, content arrives
    render('', true)
    expect(isLive()).toBe(true)

    // Each shared animation frame reveals 15% of the remaining content.
    animationTime += 33
    let state = render(content, true)
    expect(state.displayedContent.length).toBe(15)
    expect(isLive()).toBe(true)

    animationTime += 33
    state = render(content, true)
    expect(state.displayedContent.length).toBe(27)

    // Normal completion reveals the authoritative response immediately.
    state = render(content, false)
    expect(state.displayedContent).toBe(content)
    expect(state.isCatchingUp).toBe(false)
    expect(isNoop()).toBe(true)
    expect(state.showCursor).toBe(false)
  })

  test('does not reveal more than once within one animation frame', () => {
    render('', true)
    animationTime += 33

    let state = render('x'.repeat(100), true)
    expect(state.displayedContent.length).toBe(15)

    state = render('x'.repeat(120), true)
    expect(state.displayedContent.length).toBe(15)

    animationTime += 33
    state = render('x'.repeat(120), true)
    expect(state.displayedContent.length).toBe(30)
  })

  test('snaps content growth on a component that is not streaming', () => {
    const initial = 'short'
    let state = render(initial, false)
    expect(state.displayedContent).toBe(initial)
    expect(isNoop()).toBe(true)

    const grown = initial + '!'.repeat(30)
    state = render(grown, false)
    expect(state.displayedContent).toBe(grown)
    expect(isNoop()).toBe(true)
    expect(state.showCursor).toBe(false)
  })

  test('interrupt snaps to full content without subscribing', () => {
    render('', true)
    animationTime += 33
    const state = render('interrupted content', true, true)
    expect(state.displayedContent).toBe('interrupted content')
    expect(isNoop()).toBe(true)
  })
})
