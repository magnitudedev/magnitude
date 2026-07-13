/**
 * Shared timer tick store — used by useSyncExternalStore.
 *
 * Module-level singleton: one interval, many readers.
 * The interval only runs while at least one component is subscribed.
 */
let tickValue = 0
const tickListeners = new Set<() => void>()
let tickInterval: ReturnType<typeof setInterval> | null = null

function ensureInterval(): void {
  if (tickInterval) return
  tickInterval = setInterval(() => {
    tickValue++
    tickListeners.forEach((cb) => cb())
  }, 1000)
}

function stopInterval(): void {
  if (tickInterval) {
    clearInterval(tickInterval)
    tickInterval = null
  }
}

export function subscribeTick(cb: () => void): () => void {
  tickListeners.add(cb)
  ensureInterval()
  return () => {
    tickListeners.delete(cb)
    if (tickListeners.size === 0) stopInterval()
  }
}

export function getTickSnapshot(): number {
  return tickValue
}

/** No-op subscribe — when you don't want the interval running */
export function subscribeNoop(): () => void {
  return () => {}
}
