/**
 * Shared animation tick store — 80ms interval for sub-second animations.
 *
 * Module-level singleton: one interval, many readers.
 * The interval only runs while at least one component is subscribed.
 * 80ms matches the CLI's fastest current interval (braille spinner).
 * 60fps (16ms) is wasteful for terminal rendering and may overwhelm the renderer.
 */
let animationTickValue = 0
const animationTickListeners = new Set<() => void>()
let animationTickInterval: ReturnType<typeof setInterval> | null = null

function ensureInterval(): void {
  if (animationTickInterval) return
  animationTickInterval = setInterval(() => {
    animationTickValue++
    animationTickListeners.forEach((cb) => cb())
  }, 80)
}

function stopInterval(): void {
  if (animationTickInterval) {
    clearInterval(animationTickInterval)
    animationTickInterval = null
  }
}

export function subscribeAnimationTick(cb: () => void): () => void {
  animationTickListeners.add(cb)
  ensureInterval()
  return () => {
    animationTickListeners.delete(cb)
    if (animationTickListeners.size === 0) stopInterval()
  }
}

export function getAnimationTickSnapshot(): number {
  return animationTickValue
}

/** No-op subscribe — when you don't want the interval running */
export function subscribeAnimationNoop(): () => void {
  return () => {}
}

/**
 * Frozen snapshot — pair with subscribeAnimationNoop. Idle subscribers must
 * read a stable value: reading the live counter while noop-subscribed makes
 * useSyncExternalStore see a changed snapshot on prop-driven re-renders and
 * schedule a spurious extra render pass.
 */
export function getAnimationTickFrozenSnapshot(): number {
  return 0
}
