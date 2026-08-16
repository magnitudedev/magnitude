export function formatElapsedMs(elapsedMs: number): string {
  const elapsed = Math.max(0, Math.floor(elapsedMs / 1000))
  const m = Math.floor(elapsed / 60)
  const s = elapsed % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

/** Human-readable duration used by the completed-work row in chat timelines. */
export function formatWorkDuration(durationMs: number): string {
  if (durationMs < 1_000) return '<1 second'
  const totalSeconds = Math.floor(durationMs / 1_000)
  if (totalSeconds < 60) {
    return `${totalSeconds} second${totalSeconds === 1 ? '' : 's'}`
  }
  const minutes = Math.floor(totalSeconds / 60)
  const remainingSeconds = totalSeconds % 60
  if (remainingSeconds === 0) {
    return `${minutes} minute${minutes === 1 ? '' : 's'}`
  }
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}
