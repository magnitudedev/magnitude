/**
 * DaemonConnectionError — spec §10
 *
 * Full-screen overlay with a top-positioned diagnostic panel.
 * Long diagnostics render in a bounded monospace scroll area with a copy
 * affordance. Reconnecting state shows Loader2 spinner + "Reconnecting...".
 */
import {
  AlertTriangle,
  RefreshCw,
  LogOut,
  Loader2,
  Copy,
  Check,
} from "lucide-react"
import { useCallback, useState, type ReactNode } from "react"
export interface DaemonConnectionErrorProps {
  /** Error message to display */
  message: string
  /** Whether we're currently attempting to reconnect */
  reconnecting: boolean
  /** Fatal application invariant violation rather than daemon liveness */
  invariantViolation?: boolean
  /** Retry connection */
  onRetry: () => void
  /** Quit / close the app */
  onQuit: () => void
}
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Fall through to the legacy selection copy path.
  }
  try {
    const textArea = document.createElement("textarea")
    textArea.value = text
    textArea.setAttribute("readonly", "")
    textArea.style.position = "fixed"
    textArea.style.left = "-9999px"
    textArea.style.top = "0"
    document.body.appendChild(textArea)
    textArea.select()
    const copied = document.execCommand("copy")
    document.body.removeChild(textArea)
    return copied
  } catch (error) {
    console.warn("[DaemonConnectionError] Failed to copy diagnostics:", error)
    return false
  }
}
function CopyDiagnosticsButton({ text }: { text: string }): ReactNode {
  const [copied, setCopied] = useState(false)
  const [failed, setFailed] = useState(false)
  const handleCopy = useCallback(async () => {
    const ok = await copyText(text)
    setFailed(!ok)
    if (!ok) return
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1800)
  }, [text])
  const label = failed ? "Copy failed" : copied ? "Copied" : "Copy"
  return (
    <button
      onClick={handleCopy}
      className="border-slate-300 dark:border-slate-750 text-slate-600 dark:text-slate-400 hover:border-slate-400 hover:text-slate-900 dark:hover:border-slate-600 dark:hover:text-slate-200 [height:30px] inline-flex items-center [gap:6px] [padding:0_10px] [background:transparent] border border-slate-300 dark:border-slate-750 rounded-[4px] font-sans text-[13px] cursor-pointer shrink-0"
      aria-label="Copy error details"
      title="Copy error details"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
      <span>{label}</span>
    </button>
  )
}
export function DaemonConnectionError({
  message,
  reconnecting,
  invariantViolation = false,
  onRetry,
  onQuit,
}: DaemonConnectionErrorProps): ReactNode {
  const title = reconnecting
    ? "Reconnecting"
    : invariantViolation
    ? "Application error"
    : "Connection failed"
  const detailLabel = invariantViolation
    ? "Error details"
    : "Connection details"
  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="daemon-error-title"
      aria-describedby={!reconnecting ? "daemon-error-details" : undefined}
      className="fixed inset-0 z-[100] flex items-center justify-center overflow-auto bg-black/70 p-6"
    >
      <div className="[width:min(760px,_calc(100vw_-_48px))] [max-height:calc(100vh_-_48px)] bg-white dark:bg-slate-850 border border-red-600 dark:border-red-500 rounded-[6px] border-t-[3px] border-t-red-600 dark:border-t-red-500 [padding:0px] flex flex-col items-stretch text-left [box-shadow:0_16px_64px_rgba(0,0,0,0.6)] [animation:fade-in_200ms_ease-out] overflow-hidden">
        <div className="flex items-start justify-between [gap:16px] [padding:18px_20px_14px] border-b border-b-slate-200 dark:border-b-slate-800 shrink-0">
          <div className="flex items-start [gap:12px] min-w-0">
            <AlertTriangle
              size={22}
              className="text-red-600 dark:text-red-500 [margin-top:2px] shrink-0"
            />
            <div className="min-w-0">
              <div
                id="daemon-error-title"
                className="font-sans text-[17px] font-semibold text-slate-900 dark:text-slate-200 leading-[1.25]"
              >
                {title}
              </div>
              <div className="[margin-top:4px] text-slate-600 dark:text-slate-400 font-sans text-[13px] leading-[1.4]">
                {reconnecting
                  ? "Trying to restore the daemon connection."
                  : "The full diagnostic output is preserved below."}
              </div>
            </div>
          </div>
          {!reconnecting && <CopyDiagnosticsButton text={message} />}
        </div>

        <div className="[padding:16px_20px_18px] min-h-0 flex flex-col [gap:12px]">
          {reconnecting ? (
            <div className="flex items-center [gap:8px] font-sans text-[14px] text-slate-600 dark:text-slate-400">
              <Loader2
                size={16}
                className="text-blue-700 dark:text-blue-500 [animation:spin_1s_linear_infinite]"
              />
              <span>Reconnecting...</span>
            </div>
          ) : (
            <div className="min-h-0 flex flex-col [gap:6px]">
              <div className="text-red-600 dark:text-red-500 font-mono text-[12px] font-semibold [letter-spacing:0px]">
                {detailLabel}
              </div>
              <pre
                id="daemon-error-details"
                className="[max-height:min(52vh,_460px)] [min-height:96px] overflow-auto [margin:0px] [padding:12px] bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-750 rounded-[4px] text-slate-900 dark:text-slate-200 font-mono text-[12px] leading-[1.45] [letter-spacing:0px] whitespace-pre-wrap [overflow-wrap:anywhere] text-left"
              >
                {message || "No diagnostic details were provided."}
              </pre>
            </div>
          )}

          {!reconnecting && (
            <div className="flex justify-end [gap:10px] flex-wrap">
              <button
                onClick={onRetry}
                className="data-[disabled=false]:hover:opacity-90 flex items-center [gap:6px] [padding:8px_14px] bg-blue-700 dark:bg-blue-500 border-0 rounded-[4px] text-slate-900 dark:text-slate-200 font-sans text-[14px] font-medium cursor-pointer [transition:opacity_100ms]"
                data-disabled="false"
              >
                <RefreshCw size={14} />
                <span>Retry</span>
              </button>

              <button
                onClick={onQuit}
                className="border-slate-300 dark:border-slate-750 text-slate-600 dark:text-slate-400 hover:border-slate-400 hover:text-slate-900 dark:hover:border-slate-600 dark:hover:text-slate-200 flex items-center [gap:6px] [padding:8px_14px] [background:transparent] border border-slate-300 dark:border-slate-750 rounded-[4px] font-sans text-[14px] cursor-pointer [transition:all_100ms]"
              >
                <LogOut size={14} />
                <span>Quit</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
