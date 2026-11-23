import { useRef } from "react"
import { useNow } from "./shared"
import { DateTime, Option } from "effect"
import {
  type ChatAssistantMessage,
  type LivePhase,
  type LiveResponse,
  tokensPerSecond,
} from "@magnitudedev/sdk"
import { CircleNotchIcon } from "@phosphor-icons/react"

const count = (n: number) => Math.floor(n).toLocaleString()
const rate = (n: number) => n.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })
const bytes = (n: number) => (n / 1024 ** 3).toFixed(1)
const percent = (completed: number, total: number) =>
  total > 0 ? Math.min(100, Math.max(0, Math.floor(completed / total * 100))) : undefined
const elapsed = (ms: number) => {
  const seconds = Math.floor(Math.max(0, ms) / 1000)
  return String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0")
}
const phases = {
  starting: "Starting request",
  model_queued: "Waiting for model",
  preparing_model: "Preparing model",
  queued: "Waiting for inference",
  optimizing: "Optimizing model",
  loading: "Loading model",
  finalizing: "Finalizing model",
  preparing: "Preparing prompt",
  prefill: "Processing prompt",
  thinking: "Thinking",
}
const OPTIMIZING_BUDGET_MS = 60_000
const substatuses = {
  starting: "Connecting to inference",
  model_queued: "Awaiting model availability",
  preparing_model: "Preparing inference runtime",
  queued: "Awaiting execution",
  optimizing: "Testing kernel variants (one-time, up to 1 min)",
  loading: "Importing weights",
  finalizing: "Preparing model for inference",
  preparing: "Preparing conversation",
  prefill: "",
  thinking: "",
}
type Activity = keyof typeof phases
const stages = {
  queued: "model_queued",
  preparing: "preparing_model",
  optimizing: "optimizing",
  loading_weights: "loading",
  finalizing: "finalizing",
} as const

/** What a running response is waiting on, if anything: generating an answer needs no status. */
export const responseActivity = (
  response: ChatAssistantMessage,
  phase: Option.Option<LivePhase>
): Activity | undefined =>
  Option.match(phase, {
    onNone: () => "starting",
    onSome: ({ phase }) => {
      switch (phase._tag) {
        case "queued":
        case "preparing":
        case "prefill":
          return phase._tag
        case "model_loading":
          return stages[phase.stage]
        case "generating":
          return response.content.at(-1)?._tag === "reasoning" ? "thinking" : undefined
      }
    },
  })

export function ChatRequestStatus({
  response,
  live,
}: {
  response: ChatAssistantMessage
  live: Option.Option<LiveResponse>
}) {
  const now = useNow()
  const activity = responseActivity(response, Option.flatMap(live, (r) => r.phase))
  const started = useRef<{ responseId: string; activity: Activity | undefined; at: number }>({
    responseId: response.id,
    activity,
    at: DateTime.toEpochMillis(response.started),
  })
  if (started.current.responseId !== response.id) {
    started.current = { responseId: response.id, activity, at: DateTime.toEpochMillis(response.started) }
  } else if (started.current.activity !== activity) {
    started.current = { responseId: response.id, activity, at: now }
  }
  return <ChatActivityStatus response={response} live={live} now={now} activityStartedAt={started.current.at} />
}
export function ChatActivityStatus({
  response,
  live,
  now,
  activityStartedAt,
}: {
  response: ChatAssistantMessage
  live: Option.Option<LiveResponse>
  now: number
  activityStartedAt?: number
}) {
  const current = Option.flatMap(live, (r) => r.phase)
  const activity = responseActivity(response, current)
  if (!activity || Option.isSome(response.end)) return null
  const reported = Option.getOrUndefined(Option.flatMap(current, (p) => p.progress))
  const load = reported?._tag === "model_loading" ? reported.details : undefined
  const measures: string[] = []
  let progress: number | undefined
  let progressText: string | undefined
  if (activity === "optimizing" && load?.kind === "tuning") {
    measures.push(count(load.measured_configurations) + " tested")
  } else if (activity === "loading" && load?.kind === "weights") {
    progress = percent(load.completed_bytes, load.total_bytes)
    if (progress !== undefined) progressText = bytes(load.completed_bytes) + " / " + bytes(load.total_bytes) + " GB"
  } else if (activity === "prefill" && reported?._tag === "prefill") {
    progress = percent(reported.completedTokens, reported.totalTokens)
    if (progress !== undefined) progressText = count(reported.completedTokens) + " / " + count(reported.totalTokens) + " tokens"
    measures.push(count(reported.cachedTokens) + " cached")
  } else if (activity === "thinking") {
    const usage = Option.flatMap(live, (r) => r.usage)
    if (Option.isSome(usage)) measures.push(count(usage.value.completionTokens) + (usage.value.completionTokens === 1 ? " token" : " tokens"))
    const speed = Option.flatMap(live, tokensPerSecond)
    if (Option.isSome(speed)) measures.push(rate(speed.value) + " tok/s")
  }
  const phase = phases[activity]
  const substatus = substatuses[activity]
  const phaseElapsed = Math.max(0, now - (activityStartedAt ?? DateTime.toEpochMillis(response.started)))
  const timer = activity === "optimizing"
    ? `${elapsed(Math.min(phaseElapsed, OPTIMIZING_BUDGET_MS))} / ${elapsed(OPTIMIZING_BUDGET_MS)}`
    : elapsed(phaseElapsed)
  return <div className="chat-request-status mb-2">
    <div role="status" aria-label={phase} className="chat-status-line">
      <CircleNotchIcon aria-hidden="true" className="chat-status-spinner motion-safe:animate-spin" />
      <div className="chat-status-content">
        <span className="chat-status-title chat-status-shimmer">{phase}</span>
        {substatus && <span className="chat-status-work">{substatus}</span>}
        {progress !== undefined && <span className="chat-status-progress">
          <span role="progressbar" aria-label={phase} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>{progress}%</span>
          <span>{progressText}</span>
        </span>}
        {measures.map(text => <span key={text} className="chat-status-measure">{text}</span>)}
      </div>
      <time className="chat-status-elapsed" aria-label="Activity elapsed time">{timer}</time>
    </div>
  </div>
}
