/**
 * DaemonConnectionError — spec §10
 *
 * Full-screen overlay with a top-positioned diagnostic panel.
 * Long diagnostics render in a bounded monospace scroll area with a copy
 * affordance. Reconnecting state shows Loader2 spinner + "Reconnecting...".
 */
import { AlertTriangle, RefreshCw, LogOut, Loader2, Copy, Check } from "lucide-react"
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
      className="hover-outline"
      aria-label="Copy error details"
      title="Copy error details"
      style={{
        height: 30,
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "0 10px",
        background: "transparent",
        border: "1px solid var(--border-default)",
        borderRadius: 4,
        fontFamily: "var(--font-sans)",
        fontSize: 13,
        cursor: "pointer",
        flexShrink: 0,
      }}
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
  const detailLabel = invariantViolation ? "Error details" : "Connection details"

  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="daemon-error-title"
      aria-describedby={!reconnecting ? "daemon-error-details" : undefined}
      style={{
        position: "fixed",
        inset: 0,
        background: "var(--bg-overlay)",
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        overflow: "auto",
      }}
    >
      <div
        style={{
          width: "min(760px, calc(100vw - 48px))",
          maxHeight: "calc(100vh - 48px)",
          background: "var(--bg-surface)",
          border: "1px solid var(--accent-error)",
          borderRadius: 6,
          borderTop: "3px solid var(--accent-error)",
          padding: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          textAlign: "left",
          boxShadow: "0 16px 64px rgba(0,0,0,0.6)",
          animation: "fade-in 200ms ease-out",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 16,
            padding: "18px 20px 14px",
            borderBottom: "1px solid var(--border-subtle)",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12, minWidth: 0 }}>
            <AlertTriangle size={22} style={{ color: "var(--accent-error)", marginTop: 2, flexShrink: 0 }} />
            <div style={{ minWidth: 0 }}>
              <div
                id="daemon-error-title"
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 17,
                  fontWeight: 600,
                  color: "var(--fg-primary)",
                  lineHeight: 1.25,
                }}
              >
                {title}
              </div>
              <div
                style={{
                  marginTop: 4,
                  color: "var(--fg-secondary)",
                  fontFamily: "var(--font-sans)",
                  fontSize: 13,
                  lineHeight: 1.4,
                }}
              >
                {reconnecting
                  ? "Trying to restore the daemon connection."
                  : "The full diagnostic output is preserved below."}
              </div>
            </div>
          </div>
          {!reconnecting && <CopyDiagnosticsButton text={message} />}
        </div>

        <div
          style={{
            padding: "16px 20px 18px",
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {reconnecting ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontFamily: "var(--font-sans)",
                fontSize: 14,
                color: "var(--fg-secondary)",
              }}
            >
              <Loader2
                size={16}
                style={{ color: "var(--accent-primary)", animation: "spin 1s linear infinite" }}
              />
              <span>Reconnecting...</span>
            </div>
          ) : (
            <div style={{ minHeight: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              <div
                style={{
                  color: "var(--accent-error)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  fontWeight: 600,
                  letterSpacing: 0,
                }}
              >
                {detailLabel}
              </div>
              <pre
                id="daemon-error-details"
                style={{
                  maxHeight: "min(52vh, 460px)",
                  minHeight: 96,
                  overflow: "auto",
                  margin: 0,
                  padding: 12,
                  background: "var(--bg-code)",
                  border: "1px solid var(--border-default)",
                  borderRadius: 4,
                  color: "var(--fg-primary)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  lineHeight: 1.45,
                  letterSpacing: 0,
                  whiteSpace: "pre-wrap",
                  overflowWrap: "anywhere",
                  textAlign: "left",
                }}
              >
                {message || "No diagnostic details were provided."}
              </pre>
            </div>
          )}

          {!reconnecting && (
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, flexWrap: "wrap" }}>
              <button
                onClick={onRetry}
                className="hover-opacity"
                data-disabled="false"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "8px 14px",
                  background: "var(--accent-primary)",
                  border: "none",
                  borderRadius: 4,
                  color: "var(--fg-primary)",
                  fontFamily: "var(--font-sans)",
                  fontSize: 14,
                  fontWeight: 500,
                  cursor: "pointer",
                  transition: "opacity 100ms",
                }}
              >
                <RefreshCw size={14} />
                <span>Retry</span>
              </button>

              <button
                onClick={onQuit}
                className="hover-outline"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "8px 14px",
                  background: "transparent",
                  border: "1px solid var(--border-default)",
                  borderRadius: 4,
                  fontFamily: "var(--font-sans)",
                  fontSize: 14,
                  cursor: "pointer",
                  transition: "all 100ms",
                }}
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
