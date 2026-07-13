/**
 * Status indicator message — spec §9.3.6
 *
 * Plain text, --fg-secondary, font-mono. No background, border, or icon.
 */
import { type ReactNode } from "react"
import type { StatusIndicatorMessage as StatusIndicatorType } from "@magnitudedev/sdk"

export function StatusIndicator({ message }: { message: StatusIndicatorType }): ReactNode {
  return (
    <div
      style={{
        color: "var(--fg-secondary)",
        fontSize: "13px",
        fontFamily: "var(--font-mono)",
      }}
    >
      {message.message}
    </div>
  )
}
