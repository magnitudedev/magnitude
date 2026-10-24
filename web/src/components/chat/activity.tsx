import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import { Option, Stream } from "effect"
import type { ChatGeneration } from "@magnitudedev/sdk"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { number, seconds, value } from "./shared"
const tick = Atom.make(
  Stream.tick("100 millis").pipe(Stream.map(() => Date.now()))
)
const phases = {
  queued: "Waiting for model",
  optimizing: "Optimizing model",
  loading: "Loading model",
  finalizing: "Finalizing model",
  preparing: "Preparing prompt",
  prefill: "Processing prompt",
  thinking: "Thinking",
  responding: "Responding",
}
export function ChatRequestStatus({
  generation,
}: {
  generation: ChatGeneration
}) {
  const clock = useAtomValue(tick)
  const now = Result.isSuccess(clock) ? clock.value : Date.now()
  return <ChatActivityStatus generation={generation} now={now} />
}
export function ChatActivityStatus({
  generation,
  now,
}: {
  generation: ChatGeneration
  now: number
}) {
  const activity = value(generation.activity)
  if (!activity) return null
  const m = generation.metrics
  const fields: [string, string][] = []
  if (activity.phase === "prefill") {
    if (Option.isSome(activity.completed) && Option.isSome(activity.total))
      fields.push([
        "Tokens",
        `${number(activity.completed.value)} / ${number(activity.total.value)}`,
      ])
    if (Option.isSome(activity.cached))
      fields.push(["Cached", number(activity.cached.value)])
    if (Option.isSome(m.promptTokensPerSecond))
      fields.push(["Speed", `${number(m.promptTokensPerSecond.value)} tok/s`])
  } else if (activity.phase === "responding" || activity.phase === "thinking") {
    if (Option.isSome(m.outputTokens))
      fields.push(["Tokens", number(m.outputTokens.value)])
    if (Option.isSome(m.tokensPerSecond))
      fields.push(["Speed", `${number(m.tokensPerSecond.value)} tok/s`])
    if (Option.isSome(m.ttftMs)) fields.push(["TTFT", seconds(m.ttftMs.value)])
  } else if (
    activity.phase === "optimizing" &&
    Option.isSome(activity.completed) &&
    Option.isSome(activity.total)
  )
    fields.push([
      "Configurations",
      `${number(activity.completed.value)} / ${number(activity.total.value)}`,
    ])
  else if (activity.phase === "queued") fields.push(["Queue", "Waiting"])
  else if (activity.phase === "preparing") fields.push(["Stage", "Tokenizing"])
  else if (activity.phase === "finalizing")
    fields.push(["Stage", "Starting runtime"])
  fields.push(["Elapsed", seconds(Math.max(0, now - activity.since))])
  const phase = phases[activity.phase]
  return (
    <div
      role="status"
      aria-label={phase}
      className="chat-request-status mb-3 flex min-h-11 flex-wrap items-center justify-between gap-x-5 gap-y-3 px-1"
    >
      <div className="inline-flex shrink-0 items-center gap-2">
        <CircleNotchIcon
          aria-hidden="true"
          className="size-3.5 shrink-0 text-slate-500 motion-safe:animate-spin dark:text-slate-400"
        />
        <span className="chat-status-shimmer text-xs font-medium leading-5">
          {phase}
        </span>
        {["optimizing", "loading", "prefill"].includes(activity.phase) &&
          Option.isSome(activity.fraction) && (
            <span
              role="progressbar"
              aria-label={phase}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.floor(activity.fraction.value * 100)}
              className="ml-0.5 rounded bg-slate-150 px-1.5 py-0.5 text-[11px] font-medium leading-4 tabular-nums text-slate-700 dark:bg-slate-800 dark:text-slate-300"
            >
              {Math.floor(activity.fraction.value * 100)}%
            </span>
          )}
      </div>
      <dl className="chat-status-metrics flex items-center">
        {fields.map(([label, text]) => (
          <div
            key={label}
            className="border-l border-slate-200 px-3 first:border-l-0 first:pl-0 last:pr-0 dark:border-slate-750"
          >
            <dt className="text-[11px] leading-4 text-slate-500 dark:text-slate-400">
              {label}
            </dt>
            <dd className="mt-0.5 whitespace-nowrap text-xs font-medium leading-4 tabular-nums text-slate-700 dark:text-slate-300">
              {text}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
