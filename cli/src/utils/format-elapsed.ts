export function formatElapsedMs(elapsedMs: number): string {
  const elapsed = Math.max(0, Math.floor(elapsedMs / 1000))
  const m = Math.floor(elapsed / 60)
  const s = elapsed % 60
  return `${m}:${String(s).padStart(2, '0')}`
}
