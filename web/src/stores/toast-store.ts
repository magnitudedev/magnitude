/**
 * Toast store — spec §10
 *
 * Ephemeral messages with auto-dismiss. Uses a module-level store
 * with useSyncExternalStore (NO useEffect for auto-dismiss).
 *
 * Toasts auto-dismiss after 5 seconds. The timer is managed within
 * the store itself — each toast schedules its own removal via setTimeout,
 * and the timeout is stored so it can be cancelled if the toast is
 * dismissed manually before it fires.
 */

export type ToastKind = "success" | "error" | "info"

export interface ToastEntry {
  readonly id: number
  readonly kind: ToastKind
  readonly message: string
  readonly createdAt: number
}

interface InternalToast extends ToastEntry {
  timeout: ReturnType<typeof setTimeout> | null
}

// ── Module-level store ──

const DISMISS_MS = 5000
let nextId = 1
const toasts: InternalToast[] = []
const listeners = new Set<() => void>()

// Cached snapshot for useSyncExternalStore — only rebuilt when toasts change.
let snapshot: readonly ToastEntry[] = Object.freeze([])

function rebuildSnapshot(): void {
  snapshot = Object.freeze(
    toasts.map(({ timeout: _timeout, ...entry }) => entry),
  )
}

function emit(): void {
  rebuildSnapshot()
  listeners.forEach((cb) => cb())
}

function dismissInternal(id: number): void {
  const idx = toasts.findIndex((t) => t.id === id)
  if (idx === -1) return
  const toast = toasts[idx]
  if (toast.timeout) clearTimeout(toast.timeout)
  toasts.splice(idx, 1)
  emit()
}

// ── Public API ──

export function showToast(kind: ToastKind, message: string): void {
  const id = nextId++
  const timeout = setTimeout(() => {
    dismissInternal(id)
  }, DISMISS_MS)
  toasts.push({ id, kind, message, createdAt: Date.now(), timeout })
  emit()
}

export function dismissToast(id: number): void {
  dismissInternal(id)
}

// ── useSyncExternalStore interface ──

export function subscribeToast(cb: () => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function getToastSnapshot(): readonly ToastEntry[] {
  return snapshot
}
