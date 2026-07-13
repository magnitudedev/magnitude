/**
 * Toast & ToastContainer — spec §10
 *
 * Position: bottom-right of chat column, 12px from edges.
 * Background --bg-surface, border-left 3px solid message color,
 * padding 8px 12px, border-radius 0 4px 4px 0.
 * font-sans text-sm --fg-primary. Auto-dismiss 5s.
 *
 * Uses useSyncExternalStore with the toast store (NO useEffect).
 */
import { useSyncExternalStore, type ReactNode } from "react"
import { Check, AlertCircle, Info, X } from "lucide-react"
import {
  subscribeToast,
  getToastSnapshot,
  dismissToast,
  type ToastKind,
  type ToastEntry,
} from "../stores/toast-store"

// ── Toast color mapping ──

const toastBorder: Record<ToastKind, string> = {
  success: "var(--accent-success)",
  error: "var(--accent-error)",
  info: "var(--accent-info)",
}

const toastIcon: Record<ToastKind, ReactNode> = {
  success: <Check size={14} style={{ color: "var(--accent-success)" }} />,
  error: <AlertCircle size={14} style={{ color: "var(--accent-error)" }} />,
  info: <Info size={14} style={{ color: "var(--accent-info)" }} />,
}

// ── Single toast ──

function Toast({ toast }: { toast: ToastEntry }): ReactNode {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "var(--bg-surface)",
        borderLeft: `3px solid ${toastBorder[toast.kind]}`,
        padding: "8px 12px",
        borderRadius: "0 4px 4px 0",
        fontFamily: "var(--font-sans)",
        fontSize: 14,
        color: "var(--fg-primary)",
        boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
        animation: "toast-in 150ms ease-out",
        maxWidth: 320,
      }}
    >
      {toastIcon[toast.kind]}
      <span style={{ flex: 1 }}>{toast.message}</span>
      <button
        onClick={() => dismissToast(toast.id)}
        aria-label="Dismiss toast"
        style={{
          background: "transparent",
          border: "none",
          cursor: "pointer",
          padding: 0,
          display: "flex",
          color: "var(--fg-tertiary)",
        }}
      >
        <X size={14} />
      </button>
    </div>
  )
}

// ── Container ──

export function ToastContainer(): ReactNode {
  const toasts = useSyncExternalStore(subscribeToast, getToastSnapshot, getToastSnapshot)

  if (toasts.length === 0) return null

  return (
    <div
      style={{
        position: "absolute",
        bottom: 12,
        right: 12,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        zIndex: 40,
        pointerEvents: "auto",
      }}
    >
      {toasts.map((toast) => (
        <Toast key={toast.id} toast={toast} />
      ))}
    </div>
  )
}
