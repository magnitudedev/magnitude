/**
 * Ephemeral status message store (spec §3.3) — a single auto-dismissing
 * toast-style message. Module-level store with an internal timeout; readers
 * use useSyncExternalStore. Both apps produce and consume.
 */

export interface EphemeralMessage {
  readonly text: string
  readonly color?: string
  readonly tone?: "error" | "warning"
}

const DEFAULT_DISMISS_MS = 5000

let current: EphemeralMessage | null = null
let timer: ReturnType<typeof setTimeout> | null = null
const listeners = new Set<() => void>()

function notify(): void {
  for (const listener of listeners) listener()
}

export function addEphemeralMessage(text: string, color: string, dismissMs: number = DEFAULT_DISMISS_MS): void {
  if (timer) clearTimeout(timer)
  current = { text, color }
  notify()
  timer = setTimeout(() => {
    current = null
    timer = null
    notify()
  }, dismissMs)
}

export function addEphemeralLogMessage(
  text: string,
  tone: "error" | "warning",
  dismissMs: number = DEFAULT_DISMISS_MS,
): void {
  if (timer) clearTimeout(timer)
  current = { text, tone }
  notify()
  timer = setTimeout(() => {
    current = null
    timer = null
    notify()
  }, dismissMs)
}

export function clearEphemeralMessage(): void {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  if (current !== null) {
    current = null
    notify()
  }
}

export function subscribeEphemeralMessage(cb: () => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function getEphemeralMessageSnapshot(): EphemeralMessage | null {
  return current
}
