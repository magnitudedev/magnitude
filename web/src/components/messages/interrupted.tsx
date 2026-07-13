/**
 * Interrupted message — spec §9.3.9
 *
 * Standalone divider row.
 */
import { type CSSProperties, type ReactNode } from "react"
import { Square } from "lucide-react"
import type { InterruptedMessage as InterruptedType } from "@magnitudedev/sdk"

const DEFAULT_INTERRUPTED_TEXT = "Interrupted. What would you like to do instead?"

const dividerLineStyle: CSSProperties = {
  flex: "1 1 auto",
  minWidth: "32px",
  height: "1px",
  background: "var(--border-default)",
}

const leadingDividerLineStyle: CSSProperties = {
  ...dividerLineStyle,
  flex: "0 0 32px",
}

function getInterruptedText(message: InterruptedType): string {
  if (message.context === "root") {
    return DEFAULT_INTERRUPTED_TEXT
  }
  return "Agent interrupted."
}

export function InterruptedDivider({
  label = DEFAULT_INTERRUPTED_TEXT,
}: {
  label?: string
}): ReactNode {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        width: "100%",
        padding: "18px 0 18px 12px",
        fontFamily: "var(--font-sans)",
        fontSize: "12px",
        lineHeight: "16px",
        color: "var(--fg-tertiary)",
      }}
    >
      <span aria-hidden="true" style={leadingDividerLineStyle} />
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          flex: "0 1 auto",
          minWidth: 0,
        }}
      >
        <Square size={10} fill="currentColor" style={{ flexShrink: 0 }} />
        {label}
      </span>
      <span aria-hidden="true" style={dividerLineStyle} />
    </div>
  )
}

export function InterruptedMessage({ message }: { message: InterruptedType }): ReactNode {
  return <InterruptedDivider label={getInterruptedText(message)} />
}
