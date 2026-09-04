import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent"
import { Text } from "@earendil-works/pi-tui"
import { defineFSM, type StateUnion } from "@magnitudedev/utils/fsm"
import { Data, Effect, Fiber, Option, Queue, Schema } from "effect"
import type { MagnitudeObservation, MagnitudeProgress, MagnitudeTimings } from "./protocol"

export const MAGNITUDE_SUMMARY_WIDGET_KEY = "magnitude-inference-summary"
const RequestId = Schema.Number.pipe(Schema.int(), Schema.brand("ProgressRequestId"))
type RequestId = typeof RequestId.Type
type LivePhase = { readonly progress: MagnitudeProgress; readonly startedAt: number; readonly modelName: string }
class Observing extends Data.TaggedClass("Observing")<{
  readonly timings: Option.Option<MagnitudeTimings>
}> {}
class Observed extends Data.TaggedClass("Observed")<{
  readonly timings: Option.Option<MagnitudeTimings>
}> {}
class Closed extends Data.TaggedClass("Closed")<{}> {}
const requestMachine = defineFSM({ Observing, Observed, Closed }, {
  Observing: ["Observed", "Closed"], Observed: ["Closed"], Closed: [],
})
type RequestState = StateUnion<typeof requestMachine.stateClasses>
interface Run {
  readonly startedAt: number
  readonly modelName: string
  readonly requests: Map<RequestId, RequestState>
  readonly completed: Map<RequestId, MagnitudeTimings>
  readonly responses: Map<number, Option.Option<boolean>>
  readonly accepted: Set<RequestId>
}
class Idle extends Data.TaggedClass("Idle")<{}> {}
class Working extends Data.TaggedClass("Working")<{ readonly run: Run }> {}
class Settled extends Data.TaggedClass("Settled")<{ readonly run: Run; readonly settledAt: number }> {}
class Disposed extends Data.TaggedClass("Disposed")<{}> {}
const runMachine = defineFSM({ Idle, Working, Settled, Disposed }, {
  Idle: ["Working", "Disposed"], Working: ["Settled", "Idle", "Disposed"],
  Settled: ["Idle", "Disposed"], Disposed: [],
})
type RunState = StateUnion<typeof runMachine.stateClasses>

export interface ProgressRequest {
  readonly observe: (observation: MagnitudeObservation) => Effect.Effect<void>
  /** EOF closes observation, never implies successful inference. */
  readonly finish: Effect.Effect<void>
  readonly fail: Effect.Effect<void>
}
export interface ProgressTracker {
  readonly startRun: (modelName: string) => Effect.Effect<void>
  readonly beginResponse: (modelName: string) => Effect.Effect<ProgressResponse>
  readonly settleRun: Effect.Effect<void>
  readonly clear: Effect.Effect<void>
}
export interface ProgressResponse {
  readonly begin: Effect.Effect<ProgressRequest>
  /** The stock Pi parser, not observational EOF, decides the outcome. */
  readonly end: (successful: boolean) => Effect.Effect<void>
}

const seconds = (ms: number) => `${(Math.max(0, ms) / 1_000).toFixed(1)}s`
const count = (n: number) => n < 1_000 ? String(Math.round(n))
  : `${(n / (n < 1_000_000 ? 1_000 : 1_000_000)).toFixed(1).replace(/\.0$/, "")}${n < 1_000_000 ? "k" : "m"}`
const duration = (ms: number) => {
  const s = Math.floor(Math.max(0, ms) / 1_000)
  return s === 0 ? "<1 second" : s < 60 ? `${s} second${s === 1 ? "" : "s"}`
    : s % 60 === 0 ? `${s / 60} minute${s === 60 ? "" : "s"}` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}
export const formatLiveProgress = ({ progress, modelName, startedAt }: LivePhase, now: number): string | undefined => {
  switch (progress.phase) {
    case "queued": case "preparing": return undefined
    case "model_loading": return `Loading ${modelName} into memory · ${Math.round(Math.max(0, Math.min(1, progress.fraction)) * 100)}% · ${seconds(now - startedAt)}`
    case "generating": return `Working · ${seconds(now - startedAt)}`
    case "prefill": {
      const cached = Math.min(progress.cached_tokens, progress.total_tokens)
      const completed = Math.min(Math.max(progress.completed_tokens, cached), progress.total_tokens)
      return `Prefilling prompt · ${count(completed - cached)} / ${count(progress.total_tokens - cached)} tokens · ${count(cached)} cached · ${seconds(now - startedAt)}`
    }
  }
}

/** One session scope owns the timer; only observations for its active run can mutate presentation. */
export const makeProgressTracker = (
  ui: Pick<ExtensionUIContext, "setWidget" | "setWorkingMessage">,
  now: () => number = () => performance.now(),
) => Effect.gen(function* () {
  let state: RunState = new Idle()
  let nextId = 0
  let active: { readonly id: RequestId; readonly phase: LivePhase } | undefined
  let latest: RequestId | undefined
  let nextResponseId = 0
  const wake = yield* Queue.sliding<void>(1)
  const present = (f: () => void) => Effect.sync(f).pipe(Effect.catchAllCause(() => Effect.void))
  const clearSummary = present(() => ui.setWidget(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined))
  const resetRow = present(() => ui.setWorkingMessage())
  const render = present(() => { if (active) ui.setWorkingMessage(formatLiveProgress(active.phase, now())) })
  const finalize = Effect.gen(function* () {
    if (state._tag !== "Settled") return
    if ([...state.run.responses.values()].some(Option.isNone)) return
    if ([...state.run.requests.values()].some((r) => r._tag === "Observing")) return
    const settled = state
    state = runMachine.transition(settled, "Idle", {})
    const timings = [...settled.run.completed.entries()].filter(([id]) => settled.run.accepted.has(id)).sort(([a], [b]) => a - b).map(([, value]) => value)
    if ([...settled.run.responses.values()].some((outcome) => !Option.getOrElse(outcome, () => false)) || timings.length === 0) { yield* clearSummary; return }
    const tokens = timings.reduce((sum, t) => sum + t.predicted_n, 0)
    const decodeMs = timings.reduce((sum, t) => sum + t.predicted_ms, 0)
    const summary = `● ${settled.run.modelName} worked for ${duration(settled.settledAt - settled.run.startedAt)}`
      + ` · ${seconds(timings[0]!.time_to_first_token_ms)} TTFT`
      + (decodeMs > 0 && tokens > 0 ? ` · ${(tokens * 1_000 / decodeMs).toFixed(1)} tok/s` : "")
    yield* present(() => ui.setWidget(MAGNITUDE_SUMMARY_WIDGET_KEY, (_tui, theme) => new Text(theme.fg("muted", summary), 0, 0)))
  })
  const clear = Effect.gen(function* () {
    if (state._tag === "Disposed") return
    if (state._tag !== "Idle") state = runMachine.transition(state, "Idle", {})
    active = undefined
    latest = undefined
    yield* resetRow
    yield* clearSummary
  })
  const startRun = (modelName: string) => Effect.gen(function* () {
    if (state._tag === "Disposed") return
    if (state._tag === "Settled") yield* clear
    if (state._tag !== "Idle") return
    state = runMachine.transition(state, "Working", { run: { startedAt: now(), modelName, requests: new Map(), completed: new Map(), responses: new Map(), accepted: new Set<RequestId>() } })
    yield* clearSummary
  })
  const timer = yield* Effect.forever(Effect.gen(function* () {
    yield* Queue.take(wake)
    while (active) { yield* Effect.sleep("100 millis"); yield* render }
  })).pipe(Effect.forkScoped)
  yield* Effect.addFinalizer(() => Effect.gen(function* () {
    yield* clear
    if (state._tag !== "Disposed") state = runMachine.transition(state, "Disposed", {})
    yield* Fiber.interrupt(timer)
  }))
  return {
    startRun, clear,
    settleRun: Effect.gen(function* () {
      if (state._tag === "Working") state = runMachine.transition(state, "Settled", { settledAt: now() })
      yield* finalize
    }),
    beginResponse: (modelName) => Effect.gen(function* () {
      yield* startRun(modelName)
      const run = state._tag === "Working" ? state.run : undefined
      const responseId = ++nextResponseId
      const requests: RequestId[] = []
      run?.responses.set(responseId, Option.none())
      const belongs = () => run !== undefined && (state._tag === "Working" || state._tag === "Settled") && state.run === run
      const end = (successful: boolean) => Effect.gen(function* () {
        if (!belongs() || Option.isSome(run!.responses.get(responseId)!)) return
        run!.responses.set(responseId, Option.some(successful))
        const last = requests.at(-1)
        if (successful && last !== undefined) run!.accepted.add(last)
        for (const id of requests) {
          if (successful && id === last) continue
          const request = run!.requests.get(id)!
          if (request._tag !== "Closed") run!.requests.set(id, requestMachine.transition(request, "Closed", {}))
          run!.completed.delete(id)
        }
        if (!successful && latest !== undefined && requests.includes(latest)) { active = undefined; yield* resetRow; yield* clearSummary }
        yield* finalize
      })
      const begin: Effect.Effect<ProgressRequest> = Effect.gen(function* () {
        if (!belongs() || Option.isSome(run!.responses.get(responseId)!)) return { observe: () => Effect.void, finish: Effect.void, fail: Effect.void }
        const id = RequestId.make(++nextId)
        requests.push(id)
        latest = id
        active = undefined
        run?.requests.set(id, new Observing({ timings: Option.none() }))
        yield* resetRow
        const close = (failed: boolean) => Effect.gen(function* () {
          if (!belongs()) return
          const request = run!.requests.get(id)!
          if (request._tag !== "Observing") return
          if (failed) run!.requests.set(id, requestMachine.transition(request, "Closed", {}))
          else {
            run!.requests.set(id, requestMachine.transition(request, "Observed", {}))
            if (Option.isSome(request.timings)) run!.completed.set(id, request.timings.value)
          }
          if (latest === id) { active = undefined; yield* resetRow }
          yield* finalize
        })
        return {
          observe: (observation: MagnitudeObservation) => Effect.gen(function* () {
            if (!belongs()) return
            const request = run!.requests.get(id)!
            if (request._tag !== "Observing") return
            if (observation.timings) run!.requests.set(id, requestMachine.hold(request, { timings: Option.some(observation.timings) }))
            if (latest !== id || !observation.progress) return
            const progress = observation.progress
            active = progress.phase === "queued" || progress.phase === "preparing" ? undefined : {
              id, phase: { progress, modelName, startedAt: active && active.phase.progress.phase === progress.phase ? active.phase.startedAt : now() },
            }
            if (active) { yield* Queue.offer(wake, undefined); yield* render }
            else yield* resetRow
          }),
          finish: close(false), fail: close(true),
        }
      })
      return { begin, end }
    }),
  } satisfies ProgressTracker
})
