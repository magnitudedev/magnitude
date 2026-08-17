/**
 * Toast & ToastContainer — spec §10
 *
 * Position: bottom-right of chat column, 12px from edges.
 * Background --bg-surface, border-l 3px solid message color,
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

const toastBorderClass: Record<ToastKind, string> = {
  success: "border-l-green-700 dark:border-l-green-500",
  error: "border-l-red-600 dark:border-l-red-500",
  info: "border-l-blue-700 dark:border-l-blue-500",
}
const toastIcon: Record<ToastKind, ReactNode> = {
  success: <Check size={14} className="text-green-700 dark:text-green-500" />,
  error: <AlertCircle size={14} className="text-red-600 dark:text-red-500" />,
  info: <Info size={14} className="text-blue-700 dark:text-blue-500" />,
}

// ── Single toast ──

function Toast({ toast }: { toast: ToastEntry }): ReactNode {
  return (
    <div
      className={`${
        toastBorderClass[toast.kind]
      } flex max-w-80 animate-[toast-in_150ms_ease-out] items-center gap-2 rounded-r border-l-[3px] bg-white px-3 py-2 font-sans text-sm text-slate-900 shadow-xl dark:bg-slate-850 dark:text-slate-200`}
    >
      {toastIcon[toast.kind]}
      <span className="[flex:1]">{toast.message}</span>
      <button
        onClick={() => dismissToast(toast.id)}
        aria-label="Dismiss toast"
        className="[background:transparent] border-0 cursor-pointer [padding:0px] flex text-slate-500"
      >
        <X size={14} />
      </button>
    </div>
  )
}

// ── Container ──

export function ToastContainer(): ReactNode {
  const toasts = useSyncExternalStore(
    subscribeToast,
    getToastSnapshot,
    getToastSnapshot
  )
  if (toasts.length === 0) return null
  return (
    <div className="absolute [bottom:12px] [right:12px] flex flex-col [gap:8px] z-[40] pointer-events-auto">
      {toasts.map((toast) => (
        <Toast key={toast.id} toast={toast} />
      ))}
    </div>
  )
}
