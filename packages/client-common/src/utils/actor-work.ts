import type { DisplayActorWork } from '@magnitudedev/sdk'

export function isDisplayActorWorkActive(work: DisplayActorWork): boolean {
  return work.phase === 'waiting_for_model'
    || work.phase === 'working'
    || work.phase === 'waiting_for_workers'
}

export function isDisplayActorWorkClockRunning(work: DisplayActorWork): boolean {
  return isDisplayActorWorkActive(work) && work.activeSince !== null
}

export function displayActorWorkElapsedMs(work: DisplayActorWork, now: number): number {
  return isDisplayActorWorkClockRunning(work) && work.activeSince !== null
    ? work.accumulatedMs + Math.max(0, now - work.activeSince)
    : work.accumulatedMs
}
