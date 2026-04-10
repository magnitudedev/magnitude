export type WorkerTimerSnapshot = {
  state: 'working' | 'idle' | 'killed'
  activeSince: number | null
  accumulatedActiveMs: number
  resumeCount: number
}

export function computeWorkerElapsedMs(snapshot: WorkerTimerSnapshot, now: number): number {
  if (snapshot.state === 'working' && snapshot.activeSince !== null) {
    return Math.max(0, snapshot.accumulatedActiveMs + (now - snapshot.activeSince))
  }
  return Math.max(0, snapshot.accumulatedActiveMs)
}

export function formatWorkerTimer(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function isWorkerResumed(snapshot: Pick<WorkerTimerSnapshot, 'resumeCount'>): boolean {
  return snapshot.resumeCount > 0
}
