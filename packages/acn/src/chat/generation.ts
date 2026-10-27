import { Array as Arr, Cause, Clock, Data, DateTime, Duration, Effect, Option, Queue, Stream, Take } from "effect"
import type { EventSettlementError, WorkerInterruption } from "@magnitudedev/event-engine"
import {
  engineMetrics,
  type LivePhase,
  type LiveProgress,
  outcomeStatus,
  type ResponseFailure,
  type ResponseOutcome,
  type ResponseOutputPart,
  type ResponsePhase,
  type ResponseTimings,
  type ResponseUsage,
} from "@magnitudedev/acn-protocol"
import type * as Native from "@magnitudedev/icn-protocol/schemas"
import { type ChatStoreError, isChatStoreError } from "@magnitudedev/storage"
import { InferenceActivity } from "../inference-activity"
import { ServingUsage, ServingUsageRecord } from "../serving-usage"
import { ChatEngine, ResponseActivity, storageFailed } from "./engine"
import { Transcript } from "./folds"
import { ChatInference, requestRejected } from "./inference"
import { buildRequest } from "./request"

// Runs each started response: builds its request, streams it, and records what the stream shows.
// Output is recorded in groups of one part each, no longer than this, so a crash loses under this
// much of it; live viewers see every delta at once, through `ResponseActivity`.
const groupSpan = 1000

/** A moment of the response, by the monotonic clock. */
type Instant = bigint

const now: Effect.Effect<Instant> = Clock.currentTimeNanos

/** Milliseconds from one instant to a later one. */
const elapsed = (from: Instant, to: Instant) => Number(to - from) / 1_000_000

/**
 * Where a response's times are measured from: the wall clock read once, as the monotonic clock
 * stood then. The wall clock is never read again for the response, so every time it records is
 * this one plus the monotonic time since, and a wall clock stepped meanwhile cannot reorder them.
 * The command recorded the response's start by its own reading, so the anchor is never before it.
 */
interface Anchor {
  readonly wall: number
  readonly mono: Instant
}

/** The whole milliseconds of the wall clock at this instant, as the response measures it. */
const wallAt = (anchor: Anchor, instant: Instant) => anchor.wall + Math.floor(elapsed(anchor.mono, instant))

interface Group {
  readonly part: ResponseOutputPart
  readonly started: Instant
  readonly deltas: Arr.NonEmptyArray<string>
  readonly at: Arr.NonEmptyArray<number>
}

// What the stream has shown so far that the end and serving usage record.
interface Observed {
  group: Option.Option<Group>
  firstOutput: Option.Option<Instant>
  /** Whether the request reached inference: only a response that did has serving usage. */
  requested: boolean
  finishReason: Option.Option<string>
  /** The engine's latest timings report. */
  report: Option.Option<Report>
}

/** The engine's timings report, as the response records it: its counts and its durations. */
interface Report {
  readonly usage: ResponseUsage
  readonly timings: ResponseTimings
}

const phaseOf = (progress: Native.ChatCompletionProgress): ResponsePhase => {
  switch (progress.phase) {
    case "model_loading":
      return { _tag: "model_loading", stage: progress.stage }
    case "queued":
      return { _tag: "queued" }
    case "preparing":
      return { _tag: "preparing" }
    case "prefill":
      return { _tag: "prefill" }
    case "generating":
      return { _tag: "generating" }
  }
}

/** How far through its phase the response is, when the engine says. */
const progressOf = (progress: Native.ChatCompletionProgress): Option.Option<LiveProgress> => {
  switch (progress.phase) {
    case "model_loading":
      return Option.map(progress.details, (details) => ({ _tag: "model_loading", details }))
    case "prefill":
      return Option.some({
        _tag: "prefill",
        completedTokens: progress.completed_tokens,
        totalTokens: progress.total_tokens,
        cachedTokens: progress.cached_tokens,
      })
    default:
      return Option.none()
  }
}

/**
 * The engine's timings report as counts and durations. Its `prompt_n` leaves out the prompt tokens
 * found in the cache, `cache_n`, so the prompt's tokens are their sum.
 */
const reportOf = (timings: Native.Timings): Report => ({
  usage: {
    promptTokens: timings.prompt_n + timings.cache_n,
    cachedTokens: timings.cache_n,
    completionTokens: timings.predicted_n,
    draftTokens: timings.draft_n,
    draftAcceptedTokens: timings.draft_n_accepted,
  },
  timings: {
    promptMs: timings.prompt_ms,
    predictedMs: timings.predicted_ms,
    samplerMs: timings.sampler_ms,
    parserMs: timings.parser_ms,
    timeToFirstTokenMs: timings.time_to_first_token_ms,
  },
})

/** The report as a response's end and its metrics hold it: both or neither. */
const reported = (report: Option.Option<Report>) => ({
  usage: Option.map(report, ({ usage }) => usage),
  timings: Option.map(report, ({ timings }) => timings),
})

const failed = (error: ResponseFailure): ResponseOutcome => ({ _tag: "failed", error })

/** Output that could not be recorded, for the store's reason: it ends the response failed for storage. */
class OutputNotRecorded extends Data.TaggedError("OutputNotRecorded")<{
  readonly error: Extract<ResponseFailure, { readonly _tag: "storage" }>
}> {}

/** How an interruption ends the response: a stop or a shutdown, or nothing to record for a removed chat. */
const interrupted = (interruption: WorkerInterruption): Option.Option<ResponseOutcome> => {
  switch (interruption._tag) {
    case "Interrupted":
      return Option.some({ _tag: "stopped", by: "user" })
    case "Closed":
      return Option.some({ _tag: "stopped", by: "shutdown" })
    case "Removed":
      return Option.none()
  }
}

/** Output that could not be recorded ends the response: it failed, for the store's reason. */
const outputNotRecorded = (
  error: EventSettlementError | ChatStoreError
): Effect.Effect<never, OutputNotRecorded | EventSettlementError> =>
  isChatStoreError(error)
    ? Effect.fail(new OutputNotRecorded({ error: { _tag: "storage", message: storageFailed(error).message } }))
    : Effect.fail(error)

/** Generates each started response, one at a time per chat. */
export const Generation = ChatEngine.Worker.define({
  name: "Generation",
  reads: [Transcript],
  handlers: (on) => [
    // Only the stream is interruptible, so every interrupted handler still ends its response.
    on.event("response_started", ({ event: started, read, emit, update, interruption }) => Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      const inference = yield* ChatInference
      const activity = yield* InferenceActivity
      const servingUsage = yield* ServingUsage
      const { responseId } = started
      const begun: Anchor = yield* Effect.all({
        wall: Effect.map(Clock.currentTimeMillis, (wall) => Math.max(wall, DateTime.toEpochMillis(started.time))),
        mono: now,
      })
      const date = (instant: Instant) => DateTime.unsafeMake(wallAt(begun, instant))
      const observed: Observed = {
        group: Option.none(),
        firstOutput: Option.none(),
        requested: false,
        finishReason: Option.none(),
        report: Option.none(),
      }

      // The open group stays open until it is recorded, so one whose recording failed is tried again
      // at the end. Recording it is not interrupted, so a recorded group is never recorded twice.
      const flushGroup = Effect.suspend(() => Option.match(observed.group, {
        onNone: () => Effect.void,
        onSome: ({ part, started: first, deltas, at }) => Effect.uninterruptible(Effect.zipRight(
          emit({ _tag: "response_output", time: date(first), responseId, part, deltas, at }),
          Effect.sync(() => { observed.group = Option.none() })
        )),
      }))

      const output = (part: ResponseOutputPart, delta: string, received: Instant) => Effect.gen(function* () {
        if (Option.exists(observed.group, (group) => group.part !== part || elapsed(group.started, received) >= groupSpan)) {
          yield* flushGroup
        }
        const group = Option.match(observed.group, {
          onNone: (): Group => ({ part, started: received, deltas: [delta], at: [0] }),
          onSome: (open): Group => ({
            ...open,
            deltas: [...open.deltas, delta],
            at: [...open.at, Math.floor(elapsed(open.started, received))],
          }),
        })
        observed.group = Option.some(group)
        observed.firstOutput = Option.orElse(observed.firstOutput, () => Option.some(received))
        // The delta's time as its group will record it: the group's time plus the delta's offset.
        const time = DateTime.add(date(group.started), { millis: Arr.lastNonEmpty(group.at) })
        yield* update(ResponseActivity, Option.some({ _tag: "output", responseId, part: { _tag: part, text: delta, time: { started: time, ended: time } } }))
      })

      const consume = (chunk: Native.ChatCompletionChunk) => Effect.gen(function* () {
        const received = yield* now
        if (Option.isSome(chunk.progress)) {
          const phase: LivePhase = { phase: phaseOf(chunk.progress.value), progress: progressOf(chunk.progress.value) }
          yield* update(ResponseActivity, Option.some({ _tag: "phase", responseId, phase }))
        }
        if (Option.isSome(chunk.timings)) {
          const report = reportOf(chunk.timings.value)
          observed.report = Option.some(report)
          yield* update(ResponseActivity, Option.some({ _tag: "report", responseId, ...report }))
        }
        const choice = chunk.choices.find((candidate) => candidate.index === 0)
        if (choice === undefined) return
        const reasoning = Option.filter(choice.delta.reasoning_content, (text) => text.length > 0)
        if (Option.isSome(reasoning)) yield* output("reasoning", reasoning.value, received)
        const text = Option.filter(Option.flatMap(choice.delta.content, Option.fromNullable), (value) => value.length > 0)
        if (Option.isSome(text)) yield* output("text", text.value, received)
        if (Option.isSome(choice.finish_reason)) observed.finishReason = choice.finish_reason
      }).pipe(Effect.catchAll(outputNotRecorded))

      // Reads chunks as they arrive, and records an open group once its span has passed even when no
      // further chunk arrives.
      const stream = Effect.scoped(Effect.gen(function* () {
        const transcript = yield* read.get(Transcript)
        const history = Arr.takeWhile(transcript, (message) => message.id !== responseId)
        const request = yield* buildRequest(history, started).pipe(Effect.mapError(() => requestRejected))
        observed.requested = true
        const chunks = yield* Stream.toQueue(inference.generate(request))
        let streaming = true
        while (streaming) {
          const next = yield* Option.match(observed.group, {
            onNone: () => Effect.map(Queue.take(chunks), Option.some),
            onSome: (group) => Effect.flatMap(now, (current) =>
              Queue.take(chunks).pipe(Effect.timeoutOption(Duration.millis(Math.max(0, groupSpan - elapsed(group.started, current)))))),
          })
          if (Option.isNone(next)) {
            yield* flushGroup.pipe(Effect.catchAll(outputNotRecorded))
            continue
          }
          streaming = yield* Take.match(next.value, {
            onEnd: () => Effect.succeed(false),
            onFailure: Effect.failCause,
            onSuccess: (received) => Effect.as(Effect.forEach(received, consume, { discard: true }), true),
          })
        }
      }))

      // How the response ended, or none for a removed chat. A stream that ends, or fails for
      // inference or for storage, ends the response so, even with an interruption pending; a stream
      // cut short by its interruption alone ends for the reason it was interrupted. Any other
      // failure, a defect among them, fails the handler whether or not an interruption is pending.
      const observe = (streamed: typeof stream) => streamed.pipe(
        // The request asks for timings, which the engine reports with a response's first output and
        // its finish, so a finish without a report breaks the protocol.
        Effect.map(() => Option.some(Option.match(observed.finishReason, {
          onNone: () => failed({ _tag: "unreachable", message: "The inference stream ended before the response finished." }),
          onSome: (finishReason): ResponseOutcome => Option.isSome(observed.report)
            ? { _tag: "finished", finishReason }
            : failed({ _tag: "protocol", message: "The inference stream finished without its timings report." }),
        }))),
        Effect.catchTags({
          InferenceFailed: (failure) => Effect.succeed(Option.some(failed(failure.error))),
          OutputNotRecorded: (failure) => Effect.succeed(Option.some(failed(failure.error))),
        }),
        Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
          ? Effect.flatMap(interruption, Option.match({
              onNone: () => Effect.failCause(cause),
              onSome: (received) => Effect.succeed(interrupted(received)),
            }))
          : Effect.failCause(cause))
      )

      // A generation interrupted while it waited in its lane ends without streaming.
      const outcome = yield* Effect.flatMap(interruption, Option.match({
        onSome: (received) => Effect.succeed(interrupted(received)),
        onNone: () => Effect.scoped(Effect.zipRight(activity.hold, observe(restore(stream)))),
      }))
      const ended = date(yield* now)
      // The open group is recorded before the end; output that still cannot be recorded ends the
      // response failed for storage.
      const ending = yield* Effect.transposeMapOption(outcome, (decided) => flushGroup.pipe(
        Effect.catchAll(outputNotRecorded),
        Effect.as(decided),
        Effect.catchTag("OutputNotRecorded", (failure) => Effect.succeed(failed(failure.error)))
      ))
      // Serving usage records what was observed of a response that reached inference, once its
      // outcome is decided: whether or not its end is recorded, and also when its chat's removal
      // cut it short.
      const metrics = engineMetrics(reported(observed.report))
      const usage = observed.requested
        ? servingUsage.record({
            id: ServingUsageRecord.fields.id.make(`chat:${responseId}`),
            model: started.model,
            completedAt: DateTime.toEpochMillis(ended),
            input: Option.getOrNull(metrics.inputTokens),
            cached: Option.getOrNull(metrics.cachedTokens),
            output: Option.getOrNull(metrics.outputTokens),
            generationMs: Option.getOrNull(metrics.generationMs),
            firstTokenMs: Option.getOrNull(Option.map(observed.firstOutput, (first) => elapsed(begun.mono, first))),
            complete: Option.exists(ending, (decided) => ["completed", "length"].includes(outcomeStatus(decided))),
          })
        : Effect.void
      yield* Effect.transposeMapOption(ending, (decided) => emit({
        _tag: "response_ended",
        time: ended,
        responseId,
        outcome: decided,
        ...reported(observed.report),
      })).pipe(Effect.ensuring(usage))
    }))),
  ],
})
