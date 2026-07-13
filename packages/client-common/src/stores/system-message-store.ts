/**
 * System message store — shared between web, desktop, and CLI.
 *
 * Module-level store with auto-dismiss setTimeout.
 * Components read via useSyncExternalStore.
 * Both apps produce and consume system messages.
 */

export interface SystemMessage {
  readonly id: string
  readonly text: string
  readonly createdAt: number
}

let messages: SystemMessage[] = []
const listeners = new Set<() => void>()
const timers = new Map<string, ReturnType<typeof setTimeout>>()

const DEFAULT_DISMISS_MS = 5000

let idCounter = 0
function nextId(): string {
  return `sysmsg-${Date.now()}-${idCounter++}`
}

function notify(): void {
  listeners.forEach((cb) => cb())
}

/**
 * Add a system message. Auto-dismisses after `dismissMs` (default 5s).
 * If dismissMs is 0 or negative, the message persists until explicitly cleared.
 */
export function addSystemMessage(text: string, dismissMs: number = DEFAULT_DISMISS_MS): string {
  const id = nextId()
  const msg: SystemMessage = { id, text, createdAt: Date.now() }
  messages = [...messages, msg]
  notify()

  if (dismissMs > 0) {
    const timer = setTimeout(() => dismissSystemMessage(id), dismissMs)
    timers.set(id, timer)
  }
  return id
}

/** Dismiss a specific system message by id. */
export function dismissSystemMessage(id: string): void {
  messages = messages.filter((m) => m.id !== id)
  const timer = timers.get(id)
  if (timer) {
    clearTimeout(timer)
    timers.delete(id)
  }
  notify()
}

/** Clear all system messages. */
export function clearSystemMessages(): void {
  for (const timer of timers.values()) {
    clearTimeout(timer)
  }
  timers.clear()
  messages = []
  notify()
}

export function subscribeSystemMessages(cb: () => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function getSystemMessagesSnapshot(): readonly SystemMessage[] {
  return messages
}
