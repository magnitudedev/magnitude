import { Array as Arr, DateTime, Option, Schema } from "effect"
import { ModelLoadDetails } from "@magnitudedev/icn-protocol/schemas"
import { Millis, optional, tagged } from "./codec"
import { MessageId } from "./identifier"
import {
  type ResponseFailure,
  ResponseOutcome,
  ResponseReasoningEffort,
  ResponseSettings,
  ResponseTimings,
  ResponseUsage,
} from "./session-event"
import { ServingModelId } from "./serving-usage"

// A chat's transcript as folded from its session's log (spec section 4), and its running response's
// live state. These shapes are derived: a change to them is a change to the fold, never to stored
// events.

const span = Schema.Struct({ started: Millis, ended: Millis })

/** Consecutive output of one part, its deltas joined, from its first delta's time to its last's. */
export const ChatContentPart = Schema.Union(
  tagged("text", { text: Schema.String, time: span }),
  tagged("reasoning", { text: Schema.String, time: span })
)
export type ChatContentPart = typeof ChatContentPart.Type

/** Joins output to content: consecutive output of one part is one content part. */
export const appendOutput = (
  content: ReadonlyArray<ChatContentPart>,
  part: ChatContentPart
): Arr.NonEmptyReadonlyArray<ChatContentPart> => {
  const last = content.at(-1)
  return last?._tag === part._tag
    ? Arr.append(content.slice(0, -1), { ...last, text: last.text + part.text, time: { started: last.time.started, ended: part.time.ended } })
    : Arr.append(content, part)
}

export const ChatUserMessage = tagged("user", { id: MessageId, text: Schema.String, time: Millis })
export type ChatUserMessage = typeof ChatUserMessage.Type

/** How a response ended, as observed. */
export const ChatResponseEnd = Schema.Struct({
  time: Millis,
  outcome: ResponseOutcome,
  usage: optional(ResponseUsage),
  timings: optional(ResponseTimings),
})
export type ChatResponseEnd = typeof ChatResponseEnd.Type

/** The latest attempt of a response, as requested and as it ran. */
export const ChatAssistantMessage = tagged("assistant", {
  id: MessageId,
  model: ServingModelId,
  reasoningEffort: optional(ResponseReasoningEffort),
  settings: ResponseSettings,
  started: Millis,
  content: Schema.Array(ChatContentPart),
  /** Absent while the response runs. */
  end: optional(ChatResponseEnd),
})
export type ChatAssistantMessage = typeof ChatAssistantMessage.Type

export const ChatMessage = Schema.Union(ChatUserMessage, ChatAssistantMessage)
export type ChatMessage = typeof ChatMessage.Type

/** The transcript's latest response: the only one a retry may replace, and the one that may be running. */
export const latestResponse = (messages: ReadonlyArray<ChatMessage>): Option.Option<ChatAssistantMessage> =>
  Arr.findLast(messages, (message): message is ChatAssistantMessage => message._tag === "assistant")

export const ModelLoadingStage = Schema.Literal("queued", "preparing", "optimizing", "loading_weights", "finalizing")
export type ModelLoadingStage = typeof ModelLoadingStage.Type

export const ResponsePhase = Schema.Union(
  tagged("queued", {}),
  tagged("preparing", {}),
  tagged("model_loading", { stage: ModelLoadingStage }),
  tagged("prefill", {}),
  tagged("generating", {})
)
export type ResponsePhase = typeof ResponsePhase.Type

/** How far through its phase a running response is, as the engine last reported it. */
export const LiveProgress = Schema.Union(
  /** Loading the model: the kernel variants tested so far, or the weight bytes loaded. */
  tagged("model_loading", { details: ModelLoadDetails }),
  /** Processing the prompt: the tokens processed of the total, and those found in the cache. */
  tagged("prefill", {
    completedTokens: Schema.NonNegativeInt,
    totalTokens: Schema.NonNegativeInt,
    cachedTokens: Schema.NonNegativeInt,
  })
)
export type LiveProgress = typeof LiveProgress.Type

/** The phase a running response is in, and how far through it, when the engine says. */
export const LivePhase = Schema.Struct({
  phase: ResponsePhase,
  progress: optional(LiveProgress),
})
export type LivePhase = typeof LivePhase.Type

/**
 * A running response's live state, which is never stored (spec rule 0): its phase, the engine's
 * latest measurements, and the open output group, streamed but not yet in the transcript.
 */
export const LiveResponse = Schema.Struct({
  id: MessageId,
  phase: optional(LivePhase),
  /** The engine's latest timings report, both or neither. */
  usage: optional(ResponseUsage),
  timings: optional(ResponseTimings),
  pending: optional(ChatContentPart),
})
export type LiveResponse = typeof LiveResponse.Type

/**
 * `finished` with `stop` is completed and with `length` is length; any other finish is failed,
 * since chats have no tools. `stopped` by the user is stopped, and by a shutdown is interrupted.
 */
export type ResponseStatus = "running" | "completed" | "length" | "failed" | "stopped" | "interrupted"

/** The status a response ends with, by its outcome. */
export const outcomeStatus = (outcome: ResponseOutcome): Exclude<ResponseStatus, "running"> => {
  switch (outcome._tag) {
    case "finished":
      return outcome.finishReason === "stop" ? "completed" : outcome.finishReason === "length" ? "length" : "failed"
    case "failed":
      return "failed"
    case "stopped":
      return outcome.by === "user" ? "stopped" : "interrupted"
  }
}

export const responseStatus = (message: ChatAssistantMessage): ResponseStatus =>
  Option.match(message.end, { onNone: () => "running", onSome: ({ outcome }) => outcomeStatus(outcome) })

const statusLabels: { readonly [Status in ResponseStatus]: Option.Option<string> } = {
  running: Option.none(),
  completed: Option.none(),
  length: Option.some("Output limit reached"),
  failed: Option.some("Failed"),
  stopped: Option.some("Stopped"),
  interrupted: Option.some("Interrupted"),
}

/**
 * What a failure tells the reader. Each message but the inference service's is a sentence written
 * where the failure was found; the service's own message is shown as its report.
 */
const failureText = (error: ResponseFailure): string => {
  switch (error._tag) {
    case "inference":
      return `The inference service failed: ${error.message}`
    case "unreachable":
    case "rejected":
    case "storage":
    case "protocol":
      return error.message
  }
}

/** How a response's status is labelled, and what went wrong with it, if anything. */
export const responseStatusText = (
  message: ChatAssistantMessage
): { readonly label: Option.Option<string>; readonly problem: Option.Option<string> } => {
  const status = responseStatus(message)
  return {
    label: statusLabels[status],
    problem: Option.flatMap(message.end, ({ outcome }) => {
      switch (outcome._tag) {
        case "failed":
          return Option.some(failureText(outcome.error))
        case "finished":
          return status === "failed"
            ? Option.some(`The model ended with ${outcome.finishReason}. This chat has no tools.`)
            : Option.none()
        case "stopped":
          return status === "interrupted"
            ? Option.some("The response ended before completion. Retry to continue.")
            : Option.none()
      }
    }),
  }
}

/** What the engine measured of a response, from its usage and timings; absent when not reported. */
export interface EngineMetrics {
  /** The engine's measured decoding time, when it measured any. */
  readonly generationMs: Option.Option<number>
  readonly tokensPerSecond: Option.Option<number>
  readonly prefillTokensPerSecond: Option.Option<number>
  readonly inputTokens: Option.Option<number>
  readonly cachedTokens: Option.Option<number>
  readonly outputTokens: Option.Option<number>
}

/** A response's measurements, each derived from its transcript (spec section 4); absent when not observed. */
export interface ResponseMetrics extends EngineMetrics {
  /** From the response's start to its first delta of any part. */
  readonly ttftMs: Option.Option<number>
  /** From the response's start to its first answer delta. */
  readonly firstAnswerMs: Option.Option<number>
  /** The reasoning parts' durations, summed. */
  readonly thoughtForMs: Option.Option<number>
  /** From the response's start to its end. */
  readonly totalMs: Option.Option<number>
}

const between = (from: DateTime.Utc, to: DateTime.Utc) => DateTime.toEpochMillis(to) - DateTime.toEpochMillis(from)

/** Tokens a second over a measured span; none when nothing was measured. */
const rate = (tokens: number, ms: number) => (ms > 0 ? Option.some((tokens / ms) * 1000) : Option.none<number>())

/** The engine's report of a response: its usage and timings, both or neither. */
export interface EngineReport {
  readonly usage: Option.Option<ResponseUsage>
  readonly timings: Option.Option<ResponseTimings>
}

/** Output tokens a second, from the engine's report: final, or the latest while a response runs. */
export const tokensPerSecond = ({ usage, timings }: EngineReport) =>
  Option.flatMap(Option.all([usage, timings]), ([u, t]) => rate(u.completionTokens, t.predictedMs))

/** What the engine measured of a response, read from its report. */
export const engineMetrics = (report: EngineReport): EngineMetrics => {
  const { usage, timings } = report
  return {
    generationMs: Option.filter(Option.map(timings, (t) => t.predictedMs), (ms) => ms > 0),
    tokensPerSecond: tokensPerSecond(report),
    // The prompt's tokens processed, those found in the cache aside.
    prefillTokensPerSecond: Option.flatMap(Option.all([usage, timings]), ([u, t]) => rate(u.promptTokens - u.cachedTokens, t.promptMs)),
    inputTokens: Option.map(usage, (u) => u.promptTokens),
    cachedTokens: Option.map(usage, (u) => u.cachedTokens),
    outputTokens: Option.map(usage, (u) => u.completionTokens),
  }
}

/** No report: a response that ended before the engine reported, or one still running. */
const unreported: EngineReport = { usage: Option.none(), timings: Option.none() }

export const responseMetrics = ({ content, started, end }: ChatAssistantMessage): ResponseMetrics => {
  const sinceStart = (at: DateTime.Utc) => between(started, at)
  const reasoning = content.filter((part) => part._tag === "reasoning")
  return {
    ttftMs: Option.map(Option.fromNullable(content[0]), (part) => sinceStart(part.time.started)),
    firstAnswerMs: Option.map(
      Option.fromNullable(content.find((part) => part._tag === "text")),
      (part) => sinceStart(part.time.started)
    ),
    thoughtForMs: reasoning.length
      ? Option.some(reasoning.reduce((sum, part) => sum + between(part.time.started, part.time.ended), 0))
      : Option.none(),
    totalMs: Option.map(end, (ended) => sinceStart(ended.time)),
    ...engineMetrics(Option.getOrElse(end, () => unreported)),
  }
}
