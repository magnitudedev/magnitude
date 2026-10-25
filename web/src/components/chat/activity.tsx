import { Atom, Result, useAtomValue } from "@effect-atom/atom-react"
import { Option, Stream } from "effect"
import type { ChatGeneration } from "@magnitudedev/sdk"
import { CircleNotchIcon } from "@phosphor-icons/react"
import { value } from "./shared"

const tick = Atom.make(Stream.tick("1 second").pipe(Stream.map(() => Date.now())))
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
export function ChatRequestStatus({ generation }: { generation: ChatGeneration }) {
  const clock = useAtomValue(tick)
  const now = Result.isSuccess(clock) ? clock.value : Date.now()
  return <ChatActivityStatus generation={generation} now={now} />
}
export function ChatActivityStatus({ generation, now }: { generation: ChatGeneration; now: number }) {
  const activity = value(generation.activity)
  if (!activity || generation.status !== "running" || activity.phase === "generating" || activity.phase === "responding") return null
  const load = value(activity.loadDetails)
  const measures: string[] = []
  let progress: number | undefined
  let progressText: string | undefined
  const substatus = {
    starting: "Connecting to inference",
    model_queued: "Awaiting model availability",
    preparing_model: "Preparing inference runtime",
    queued: "Awaiting execution",
    optimizing: "Testing kernel variants",
    loading: "Importing weights",
    finalizing: "Preparing model for inference",
    preparing: "Preparing conversation",
    prefill: "",
    thinking: "",
  }[activity.phase]
  if (activity.phase === "optimizing" && load?.kind === "tuning") {
    measures.push(count(load.measured_configurations) + " tested")
  } else if (activity.phase === "loading" && load?.kind === "weights") {
    progress = percent(load.completed_bytes, load.total_bytes)
    if (progress !== undefined) progressText = bytes(load.completed_bytes) + " / " + bytes(load.total_bytes) + " GB"
  } else if (activity.phase === "prefill") {
    const completed = value(activity.completed), total = value(activity.total)
    if (completed !== undefined && total !== undefined) {
      progress = percent(completed, total)
      if (progress !== undefined) progressText = count(completed) + " / " + count(total) + " tokens"
    }
    if (Option.isSome(activity.cached)) measures.push(count(activity.cached.value) + " cached")
  } else if (activity.phase === "thinking") {
    const m = generation.metrics
    if (Option.isSome(m.outputTokens)) measures.push(count(m.outputTokens.value) + " tokens")
    if (Option.isSome(m.tokensPerSecond)) measures.push(rate(m.tokensPerSecond.value) + " tok/s")
  }
  const phase = phases[activity.phase]
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
      <time className="chat-status-elapsed" aria-label="Request elapsed time">{elapsed(now - generation.createdAt)}</time>
    </div>
  </div>
}
