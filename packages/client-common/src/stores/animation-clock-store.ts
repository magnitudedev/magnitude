/** Shared 30 FPS clock for terminal animation presentation. */
const FRAME_INTERVAL_MS = 1_000 / 30

let animationTimeMs = 0
let lastFrameAt: number | null = null
const listeners = new Set<() => void>()
let interval: ReturnType<typeof setInterval> | null = null

function now(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now()
}

function ensureInterval(): void {
  if (interval !== null) return
  lastFrameAt = now()
  interval = setInterval(() => {
    const frameAt = now()
    animationTimeMs += Math.max(0, frameAt - (lastFrameAt ?? frameAt))
    lastFrameAt = frameAt
    listeners.forEach((listener) => listener())
  }, FRAME_INTERVAL_MS)
}

function stopInterval(): void {
  if (interval === null) return
  clearInterval(interval)
  interval = null
  lastFrameAt = null
}

export function subscribeAnimationClock(listener: () => void): () => void {
  listeners.add(listener)
  ensureInterval()
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) stopInterval()
  }
}

export function getAnimationTimeSnapshot(): number {
  return animationTimeMs
}

export function subscribeAnimationNoop(): () => void {
  return () => {}
}

export function getAnimationTimeFrozenSnapshot(): number {
  return 0
}

export function animationStep(timeMs: number, stepDurationMs: number): number {
  if (!Number.isFinite(stepDurationMs) || stepDurationMs <= 0) return 0
  return Math.floor(timeMs / stepDurationMs)
}

/** Smooth 0→1→0 pulse with zero slope at both endpoints. */
export function animationPulse(timeMs: number, durationMs: number): number {
  if (!Number.isFinite(durationMs) || durationMs <= 0) return 0
  const phase = ((timeMs % durationMs) + durationMs) % durationMs / durationMs
  return 0.5 - 0.5 * Math.cos(phase * Math.PI * 2)
}
