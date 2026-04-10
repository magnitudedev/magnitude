import { describe, expect, test } from 'bun:test'
import { computeWorkerElapsedMs, formatWorkerTimer, isWorkerResumed } from './task-list-worker-timer'

describe('task-list-worker-timer', () => {
  test('running elapsed uses accumulated plus current stint', () => {
    const elapsed = computeWorkerElapsedMs(
      { state: 'working', activeSince: 1_000, accumulatedActiveMs: 20_000, resumeCount: 0 },
      6_000,
    )
    expect(elapsed).toBe(25_000)
  })

  test('idle elapsed uses accumulated only', () => {
    const elapsed = computeWorkerElapsedMs(
      { state: 'idle', activeSince: 1_000, accumulatedActiveMs: 20_000, resumeCount: 0 },
      6_000,
    )
    expect(elapsed).toBe(20_000)
  })

  test('formats m:ss', () => {
    expect(formatWorkerTimer(83_000)).toBe('1:23')
  })

  test('detects resumed from resumeCount', () => {
    expect(isWorkerResumed({ resumeCount: 0 })).toBe(false)
    expect(isWorkerResumed({ resumeCount: 1 })).toBe(true)
  })
})
