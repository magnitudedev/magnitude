import { describe, expect, it } from 'vitest'
import type { DisplayActorWork } from '@magnitudedev/sdk'
import {
  displayActorWorkElapsedMs,
  displayActorWorkLiveState,
  isDisplayActorWorkActive,
  isDisplayActorWorkClockRunning,
} from './actor-work'

const work = (overrides: Partial<DisplayActorWork> = {}): DisplayActorWork => ({
  phase: 'working',
  activeSince: 1_000,
  lastWorkMs: 0,
  accumulatedMs: 0,
  resumeCount: 0,
  activity: null,
  activeChildCount: 0,
  ...overrides,
})

describe('display actor work timing', () => {
  it('adds the current interval to previously accumulated work', () => {
    expect(displayActorWorkElapsedMs(work({
      activeSince: 4_000,
      accumulatedMs: 3_000,
    }), 6_000)).toBe(5_000)
  })

  it('keeps accumulated work paused during a model-only wait', () => {
    const waiting = work({
      phase: 'waiting_for_model',
      activeSince: null,
      accumulatedMs: 3_000,
    })

    expect(isDisplayActorWorkActive(waiting)).toBe(true)
    expect(isDisplayActorWorkClockRunning(waiting)).toBe(false)
    expect(displayActorWorkLiveState(waiting)).toBe('waiting_for_model')
    expect(displayActorWorkElapsedMs(waiting, 10_000)).toBe(3_000)
  })

  it('advances a model wait while a worker keeps the interval open', () => {
    const waiting = work({
      phase: 'waiting_for_model',
      activeSince: 8_000,
      accumulatedMs: 3_000,
      activeChildCount: 1,
    })

    expect(isDisplayActorWorkClockRunning(waiting)).toBe(true)
    expect(displayActorWorkLiveState(waiting)).toBe('working')
    expect(displayActorWorkElapsedMs(waiting, 10_000)).toBe(5_000)
  })

  it('maps terminal work to an inactive live presentation', () => {
    expect(displayActorWorkLiveState(work({
      phase: 'worked',
      activeSince: null,
      lastWorkMs: 5_000,
    }))).toBe('inactive')
  })
})
