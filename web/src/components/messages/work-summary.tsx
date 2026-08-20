import { type ReactNode } from "react"
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
        onSome: (tokensPerSecond) => ` ${tokensPerSecond.toFixed(1)} tok/s`,
      }),
  })
}
export function WorkSummary({
  message,
}: {
  message: WorkSummaryMessage
}): ReactNode {
  const speed = Option.flatMap(message.performance, (performance) =>
    performance.decodeTokensPerSecond)
  const summary = Option.match(message.performance, {
    onNone: () => `Worked for ${formatWorkDuration(message.durationMs)}`,
    onSome: (performance) =>
      `${performance.modelDisplayName} worked for ${formatWorkDuration(message.durationMs)}`,
  })
  return (
    <div
      data-message-type="work-summary"
      className="flex min-h-[18px] items-center gap-3 font-sans text-[12px] leading-[18px] text-slate-500"
    >
      <span>{summary}</span>
      {Option.isSome(speed) ? (
        <span className="tabular-nums text-slate-400 dark:text-slate-500">
          {speed.value.toFixed(1)} tok/s
        </span>
      ) : null}
    </div>
  )
}
