export type WorkerTimerState = {
  state: 'spawning' | 'working' | 'idle' | 'killing' | 'killed-ghost'
  activeSince: number | null
  accumulatedActiveMs: number
  resumeCount: number
}

export function computeWorkerElapsedMs(state: WorkerTimerState, now: number): number {
  if (state.state === 'working' && state.activeSince !== null) {
    return Math.max(0, state.accumulatedActiveMs + (now - state.activeSince))
  }
  return Math.max(0, state.accumulatedActiveMs)
}

export function formatWorkerTimer(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function isWorkerResumed(state: Pick<WorkerTimerState, 'resumeCount'>): boolean {
  return state.resumeCount > 0
}
