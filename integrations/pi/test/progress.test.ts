import { Effect, Exit, Scope } from "effect"
import { describe, expect, it, vi } from "vitest"
import { MAGNITUDE_SUMMARY_WIDGET_KEY, makeProgressTracker, formatLiveProgress } from "../extensions/progress"

const timings = (predicted_n = 10, predicted_ms = 200) => ({
  prompt_ms: 500, time_to_first_token_ms: 3_400, predicted_n, predicted_ms, predicted_per_second: predicted_n * 1_000 / predicted_ms,
})
const testTracker = (test: (fixture: { tracker: Effect.Effect.Success<ReturnType<typeof makeProgressTracker>>; setWidget: ReturnType<typeof vi.fn>; setWorkingMessage: ReturnType<typeof vi.fn>; advance: (ms: number) => void }) => Effect.Effect<void>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    let now = 0
    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    const tracker = yield* makeProgressTracker({ setWidget, setWorkingMessage }, () => now)
    yield* test({ tracker, setWidget, setWorkingMessage, advance: (ms) => { now += ms } })
  })))
const summary = (widget: ReturnType<typeof vi.fn>) => {
  const factory = widget.mock.calls.at(-1)?.[1]
  if (!factory) return undefined
  return factory({}, { fg: (_color: string, text: string) => text }).render(200).map((line: string) => line.trimEnd()).join("\n")
}

describe("scoped progress lifecycle", () => {
  it("formats the approved live phases and clamps inconsistent counters", () => {
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "model_loading", fraction: 0.47 } }, 2300)).toBe("Loading Model into memory · 47% · 2.3s")
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "preparing" } }, 2300)).toBeUndefined()
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "prefill", completed_tokens: 14020, total_tokens: 14300, cached_tokens: 13200 } }, 400)).toBe("Prefilling prompt · 820 / 1.1k tokens · 13.2k cached · 0.4s")
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "prefill", completed_tokens: 9000, total_tokens: 1000, cached_tokens: 2000 } }, 0)).toBe("Prefilling prompt · 0 / 0 tokens · 1k cached · 0.0s")
  })

  it("waits for semantic success and delayed observers, using cumulative timings once", () => testTracker(({ tracker, setWidget, advance }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const request = yield* response.begin
    yield* request.observe({ timings: timings(5, 100) })
    yield* request.observe({ timings: timings(10, 200) })
    advance(6000)
    yield* response.end(true)
    yield* tracker.settleRun
    expect(summary(setWidget)).toBeUndefined()
    yield* request.finish
    expect(summary(setWidget)).toBe("● Model worked for 6 seconds · 3.4s TTFT · 50.0 tok/s")
    yield* request.observe({ timings: timings(900, 1) })
    yield* request.fail
    expect(summary(setWidget)).toBe("● Model worked for 6 seconds · 3.4s TTFT · 50.0 tok/s")
  })))

  it("never reports success for EOF followed by Pi error", () => testTracker(({ tracker, setWidget }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const request = yield* response.begin
    yield* request.observe({ timings: timings() })
    yield* request.finish
    yield* tracker.settleRun
    expect(summary(setWidget)).toBeUndefined()
    yield* response.end(false)
    expect(summary(setWidget)).toBeUndefined()
    yield* request.observe({ progress: { phase: "generating" }, timings: timings() })
    yield* response.end(true)
    expect(summary(setWidget)).toBeUndefined()
  })))

  it("accounts for overlapping responses independent of observer completion order", () => testTracker(({ tracker, setWidget, setWorkingMessage, advance }) => Effect.gen(function* () {
    const first = yield* tracker.beginResponse("Model")
    const a = yield* first.begin
    const second = yield* tracker.beginResponse("Model")
    const b = yield* second.begin
    yield* b.observe({ progress: { phase: "generating" }, timings: timings(20, 200) })
    const latestRow = setWorkingMessage.mock.calls.at(-1)
    yield* a.observe({ progress: { phase: "prefill", completed_tokens: 0, total_tokens: 20, cached_tokens: 0 }, timings: timings() })
    yield* a.finish
    expect(setWorkingMessage.mock.calls.at(-1)).toEqual(latestRow)
    yield* b.finish
    yield* second.end(true)
    yield* first.end(true)
    advance(6000)
    yield* tracker.settleRun
    expect(summary(setWidget)).toBe("● Model worked for 6 seconds · 3.4s TTFT · 75.0 tok/s")
  })))

  it("does not count failed HTTP attempts in a successful retry", () => testTracker(({ tracker, setWidget }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const failed = yield* response.begin
    yield* failed.observe({ timings: timings(900, 1) })
    yield* failed.finish
    const retried = yield* response.begin
    yield* retried.observe({ timings: timings() })
    yield* response.end(true)
    yield* retried.finish
    yield* tracker.settleRun
    expect(summary(setWidget)).toBe("● Model worked for <1 second · 3.4s TTFT · 50.0 tok/s")
  })))

  it("ignores observations from cleared runs and terminal requests", () => testTracker(({ tracker, setWidget, setWorkingMessage }) => Effect.gen(function* () {
    const old = yield* tracker.beginResponse("Old")
    const request = yield* old.begin
    yield* request.fail
    yield* request.observe({ progress: { phase: "generating" } })
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    yield* tracker.clear
    const current = yield* tracker.beginResponse("Current")
    const fresh = yield* current.begin
    yield* fresh.observe({ progress: { phase: "generating" } })
    const row = setWorkingMessage.mock.calls.at(-1)
    yield* old.end(false)
    yield* request.finish
    expect(setWorkingMessage.mock.calls.at(-1)).toEqual(row)
    expect(setWidget).toHaveBeenLastCalledWith(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
  })))

  it("cleans presentation and prevents late work after its scope closes", async () => {
    const scope = Effect.runSync(Scope.make())
    const ui = { setWidget: vi.fn(), setWorkingMessage: vi.fn() }
    const tracker = await Effect.runPromise(makeProgressTracker(ui).pipe(Scope.extend(scope)))
    const response = Effect.runSync(tracker.beginResponse("Model"))
    const request = Effect.runSync(response.begin)
    Effect.runSync(request.observe({ progress: { phase: "generating" } }))
    await Effect.runPromise(Scope.close(scope, Exit.void))
    const count = ui.setWorkingMessage.mock.calls.length
    Effect.runSync(request.observe({ progress: { phase: "generating" } }))
    Effect.runSync(response.end(true))
    const late = Effect.runSync(tracker.beginResponse("Model"))
    Effect.runSync(late.begin)
    expect(ui.setWorkingMessage).toHaveBeenCalledTimes(count)
  })

  it("isolates throwing UI callbacks", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const tracker = yield* makeProgressTracker({ setWidget: () => { throw Error("widget") }, setWorkingMessage: () => { throw Error("row") } })
      const response = yield* tracker.beginResponse("Model")
      const request = yield* response.begin
      yield* request.observe({ progress: { phase: "generating" }, timings: timings() })
      yield* request.finish
      yield* response.end(true)
      yield* tracker.settleRun
    })))
  })
})
