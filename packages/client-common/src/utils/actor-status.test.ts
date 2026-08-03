import { describe, expect, it } from 'vitest'
import type { DisplayRootStatus, DisplayWorkerStatus } from '@magnitudedev/sdk'
import {
  displayRootStatusElapsedMs,
  displayWorkerStatusElapsedMs,
  isDisplayRootStatusActive,
  isDisplayWorkerStatusClockRunning,
} from './actor-status'

const workerStatus = (
  overrides: Partial<DisplayWorkerStatus> = {},
): DisplayWorkerStatus => ({
  phase: 'working',
  activeSince: 1_000,
  lastWorkMs: 0,
  accumulatedMs: 0,
  resumeCount: 0,
  ...overrides,
})

describe('display actor status timing', () => {
  it('measures root chain wall time from turn start', () => {
    const status: Extract<DisplayRootStatus, { readonly _tag: 'Working' }> = {
      _tag: 'Working',
      chainStartedAt: 1_000,
      detail: { _tag: 'NoDetail' },
      activeChildCount: 0,
    }

    expect(isDisplayRootStatusActive(status)).toBe(true)
    expect(displayRootStatusElapsedMs(status, 6_000)).toBe(5_000)
  })

  it('adds the current worker interval to previously accumulated time', () => {
    expect(displayWorkerStatusElapsedMs(workerStatus({
      activeSince: 4_000,
      accumulatedMs: 3_000,
    }), 6_000)).toBe(5_000)
  })

  it('uses the last completed stint when the worker clock is stopped', () => {
    const status = workerStatus({ phase: 'worked', activeSince: null, lastWorkMs: 5_000 })
    expect(isDisplayWorkerStatusClockRunning(status)).toBe(false)
    expect(displayWorkerStatusElapsedMs(status, 10_000)).toBe(5_000)
  })
})
