/**
 * Copy-feedback hook — shared between web, desktop, and CLI.
 *
 * Returns `{ copied, triggerCopy }`. `triggerCopy` sets `copied=true`
 * and starts a timeout that resets to `false` after `resetMs` (default 2s).
 * The timeout lives in a module-level store keyed by instance id.
 * No useEffect — useRef allocates a stable instance id (non-reactive guard,
 * permitted per design rule m3).
 */
import { useSyncExternalStore, useRef } from "react"

const DEFAULT_RESET_MS = 2000

interface CopyFeedbackState {
  copied: boolean
  timer: ReturnType<typeof setTimeout> | null
}

const states = new Map<number, CopyFeedbackState>()
const listeners = new Map<number, Set<() => void>>()
let idCounter = 0

function getState(id: number): CopyFeedbackState {
  return states.get(id) ?? { copied: false, timer: null }
}

function setState(id: number, state: CopyFeedbackState): void {
  states.set(id, state)
  listeners.get(id)?.forEach((cb) => cb())
}

function subscribe(id: number, cb: () => void): () => void {
  if (!listeners.has(id)) {
    listeners.set(id, new Set())
  }
  listeners.get(id)!.add(cb)
  return () => {
    listeners.get(id)?.delete(cb)
  }
}

/**
 * Copy-feedback hook. Call once per component instance.
 * Returns `{ copied, triggerCopy }`.
 */
export function useCopyFeedback(resetMs: number = DEFAULT_RESET_MS): {
  copied: boolean
  triggerCopy: () => void
} {
  const idRef = useRef<number>(++idCounter)
  const id = idRef.current

  const copied = useSyncExternalStore(
    (cb) => subscribe(id, cb),
    () => getState(id).copied,
    () => false,
  )

  function triggerCopy(): void {
    const current = getState(id)
    if (current.timer) clearTimeout(current.timer)
    const timer = setTimeout(() => {
      setState(id, { copied: false, timer: null })
    }, resetMs)
    setState(id, { copied: true, timer })
  }

  return { copied, triggerCopy }
}
