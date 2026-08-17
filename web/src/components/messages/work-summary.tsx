import { type ReactNode } from "react"
import { Circle } from "lucide-react"
import { Option } from "effect"
import { formatWorkDuration } from "@magnitudedev/client-common"
import type { WorkSummaryMessage } from "@magnitudedev/sdk"
export function workSummaryLabel(message: WorkSummaryMessage): string {
  return Option.match(message.performance, {
    onNone: () => `Worked for ${formatWorkDuration(message.durationMs)}`,
    onSome: (performance) =>
      `${performance.modelDisplayName} worked for ${formatWorkDuration(
        message.durationMs
      )}` +
      Option.match(performance.decodeTokensPerSecond, {
        onNone: () => "",
        onSome: (tokensPerSecond) => ` · ${tokensPerSecond.toFixed(1)} tok/s`,
      }),
  })
}
export function WorkSummary({
  message,
}: {
  message: WorkSummaryMessage
}): ReactNode {
  return (
    <div
      data-message-type="work-summary"
      className="flex items-center [gap:7px] [min-height:18px] text-slate-500 font-sans text-[12px] leading-[18px]"
    >
      <Circle
        size={7}
        fill="currentColor"
        aria-hidden="true"
        className="shrink-0"
      />
      <span>{workSummaryLabel(message)}</span>
    </div>
  )
}
