import type { DisplayRootStatus, DisplayWorkerStatus } from '@magnitudedev/sdk'

export function isDisplayRootStatusActive(
  status: DisplayRootStatus,
): status is Extract<DisplayRootStatus, { readonly _tag: 'Working' }> {
  return status._tag === 'Working'
}

export function displayRootStatusElapsedMs(
  status: Extract<DisplayRootStatus, { readonly _tag: 'Working' }>,
  now: number,
): number {
  return Math.max(0, now - status.chainStartedAt)
}

export function isDisplayWorkerStatusClockRunning(status: DisplayWorkerStatus): boolean {
  return status.phase === 'working' && status.activeSince !== null
}

export function displayWorkerStatusElapsedMs(status: DisplayWorkerStatus, now: number): number {
  return isDisplayWorkerStatusClockRunning(status) && status.activeSince !== null
    ? status.accumulatedMs + Math.max(0, now - status.activeSince)
    : status.lastWorkMs
}
