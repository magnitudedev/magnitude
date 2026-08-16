import { type ReactNode } from "react"
import { Circle } from "lucide-react"
import { Option } from "effect"
import { formatWorkDuration } from "@magnitudedev/client-common"
import type { WorkSummaryMessage } from "@magnitudedev/sdk"

export function workSummaryLabel(message: WorkSummaryMessage): string {
  return Option.match(message.performance, {
    onNone: () => `Worked for ${formatWorkDuration(message.durationMs)}`,
    onSome: (performance) =>
      `${performance.modelDisplayName} worked for ${formatWorkDuration(message.durationMs)}`
      + Option.match(performance.decodeTokensPerSecond, {
        onNone: () => "",
        onSome: (tokensPerSecond) => ` · ${tokensPerSecond.toFixed(1)} tok/s`,
      }),
  })
}

export function WorkSummary({ message }: { message: WorkSummaryMessage }): ReactNode {
  return (
    <div
      data-message-type="work-summary"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        minHeight: 18,
        color: "var(--fg-tertiary)",
        fontFamily: "var(--font-sans)",
        fontSize: 12,
        lineHeight: "18px",
      }}
    >
      <Circle size={7} fill="currentColor" aria-hidden="true" style={{ flexShrink: 0 }} />
      <span>{workSummaryLabel(message)}</span>
    </div>
  )
}
