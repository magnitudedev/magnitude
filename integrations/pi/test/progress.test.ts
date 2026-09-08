import { Effect, Exit, Scope } from "effect"
import { describe, expect, it, vi } from "vitest"
import { makeProgressTracker, formatLiveProgress, formatElapsed, formatSummary } from "../extensions/progress"

const timings = (predicted_n = 10, predicted_ms = 200) => ({
  prompt_ms: 500, time_to_first_token_ms: 3_400, predicted_n, predicted_ms, predicted_per_second: predicted_n * 1_000 / predicted_ms,
})
const testTracker = (test: (fixture: { tracker: Effect.Effect.Success<ReturnType<typeof makeProgressTracker>>; appendSummary: ReturnType<typeof vi.fn>; setWorkingMessage: ReturnType<typeof vi.fn>; advance: (ms: number) => void }) => Effect.Effect<void>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    let now = 0
    const appendSummary = vi.fn()
    const setWorkingMessage = vi.fn()
    const tracker = yield* makeProgressTracker({ setWorkingMessage }, appendSummary, () => now)
    yield* test({ tracker, appendSummary, setWorkingMessage, advance: (ms) => { now += ms } })
  })))
const summary = (append: ReturnType<typeof vi.fn>) => {
  const data = append.mock.calls.at(-1)?.[0]
  return data ? formatSummary(data) : undefined
}

describe("scoped progress lifecycle", () => {
  it.each([
    [0, "<1s"], [999, "<1s"], [1000, "1s"], [38000, "38s"],
    [59999, "59s"], [60000, "1m 0s"], [61000, "1m 1s"], [65000, "1m 5s"], [72000, "1m 12s"], [3600000, "60m 0s"],
  ])("formats completed work at %s ms as %s", (elapsedMs, expected) => {
    expect(formatSummary({ modelName: "Model", elapsedMs, ttftMs: 500, generatedTokens: 0, decodeMs: 0 }))
      .toBe(`● Model worked for ${expected} · 0.5s TTFT`)
  })
  it.each([
    [-1, "0s"], [0, "0s"], [999, "0s"], [1000, "1s"],
    [59999, "59s"], [60000, "1m 0s"], [107000, "1m 47s"], [237999, "3m 57s"],
    [3600000, "60m 0s"], [6000000, "100m 0s"],
  ])("formats %s ms as %s without decimals or wrapping", (ms, expected) => {
    expect(formatElapsed(ms)).toBe(expected)
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "generating" } }, ms)).toBe(`Working · ${expected}`)
  })

  it("appends once per successful run and retains history across new runs and clearing", () => testTracker(({ tracker, appendSummary, advance }) => Effect.gen(function* () {
    for (const modelName of ["First", "Second"]) {
      const response = yield* tracker.beginResponse(modelName)
      const request = yield* response.begin
      yield* request.observe({ timings: timings() })
      advance(107000)
      yield* request.finish
      yield* response.end(true)
      yield* tracker.settleRun
      yield* tracker.settleRun
      yield* request.finish
      yield* response.end(true)
      yield* tracker.clear
    }
    expect(appendSummary).toHaveBeenCalledTimes(2)
    expect(appendSummary.mock.calls.map(([data]) => formatSummary(data))).toEqual([
      "● First worked for 1m 47s · 3.4s TTFT · 50.0 tok/s",
      "● Second worked for 1m 47s · 3.4s TTFT · 50.0 tok/s",
    ])
    const cancelled = yield* tracker.beginResponse("Cancelled")
    const request = yield* cancelled.begin
    yield* request.observe({ timings: timings() })
    yield* request.finish
    yield* cancelled.end(false)
    yield* tracker.settleRun
    expect(appendSummary).toHaveBeenCalledTimes(2)
  })))

  it("omits throughput when no decode duration is available", () => {
    expect(formatSummary({ modelName: "Model", elapsedMs: 1000, ttftMs: 500, generatedTokens: 0, decodeMs: 0 }))
      .toBe("● Model worked for 1s · 0.5s TTFT")
  })
  it("formats the approved live phases and clamps inconsistent counters", () => {
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "model_loading", fraction: 0.47 } }, 2300)).toBe("Loading Model into memory · 47% · 2s")
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "preparing" } }, 2300)).toBeUndefined()
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "prefill", completed_tokens: 14020, total_tokens: 14300, cached_tokens: 13200 } }, 400)).toBe("Prefilling prompt · 820 / 1.1k tokens · 13.2k cached · 0s")
    expect(formatLiveProgress({ modelName: "Model", startedAt: 0, progress: { phase: "prefill", completed_tokens: 9000, total_tokens: 1000, cached_tokens: 2000 } }, 0)).toBe("Prefilling prompt · 0 / 0 tokens · 1k cached · 0s")
  })

  it("waits for semantic success and delayed observers, using cumulative timings once", () => testTracker(({ tracker, appendSummary, advance }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const request = yield* response.begin
    yield* request.observe({ timings: timings(5, 100) })
    yield* request.observe({ timings: timings(10, 200) })
    advance(6000)
    yield* response.end(true)
    yield* tracker.settleRun
    expect(summary(appendSummary)).toBeUndefined()
    yield* request.finish
    expect(summary(appendSummary)).toBe("● Model worked for 6s · 3.4s TTFT · 50.0 tok/s")
    yield* request.observe({ timings: timings(900, 1) })
    yield* request.fail
    expect(summary(appendSummary)).toBe("● Model worked for 6s · 3.4s TTFT · 50.0 tok/s")
  })))

  it("never reports success for EOF followed by Pi error", () => testTracker(({ tracker, appendSummary }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const request = yield* response.begin
    yield* request.observe({ timings: timings() })
    yield* request.finish
    yield* tracker.settleRun
    expect(summary(appendSummary)).toBeUndefined()
    yield* response.end(false)
    expect(summary(appendSummary)).toBeUndefined()
    yield* request.observe({ progress: { phase: "generating" }, timings: timings() })
    yield* response.end(true)
    expect(summary(appendSummary)).toBeUndefined()
  })))

  it("accounts for overlapping responses independent of observer completion order", () => testTracker(({ tracker, appendSummary, setWorkingMessage, advance }) => Effect.gen(function* () {
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
    expect(summary(appendSummary)).toBe("● Model worked for 6s · 3.4s TTFT · 75.0 tok/s")
  })))

  it("does not count failed HTTP attempts in a successful retry", () => testTracker(({ tracker, appendSummary }) => Effect.gen(function* () {
    const response = yield* tracker.beginResponse("Model")
    const failed = yield* response.begin
    yield* failed.observe({ timings: timings(900, 1) })
    yield* failed.finish
    const retried = yield* response.begin
    yield* retried.observe({ timings: timings() })
    yield* response.end(true)
    yield* retried.finish
    yield* tracker.settleRun
    expect(summary(appendSummary)).toBe("● Model worked for <1s · 3.4s TTFT · 50.0 tok/s")
  })))

  it("ignores observations from cleared runs and terminal requests", () => testTracker(({ tracker, appendSummary, setWorkingMessage }) => Effect.gen(function* () {
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
    expect(appendSummary).not.toHaveBeenCalled()
  })))

  it("cleans presentation and prevents late work after its scope closes", async () => {
    const scope = Effect.runSync(Scope.make())
    const ui = { appendSummary: vi.fn(), setWorkingMessage: vi.fn() }
    const tracker = await Effect.runPromise(makeProgressTracker(ui, ui.appendSummary).pipe(Scope.extend(scope)))
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
      const tracker = yield* makeProgressTracker({ setWorkingMessage: () => { throw Error("row") } }, () => { throw Error("append") })
      const response = yield* tracker.beginResponse("Model")
      const request = yield* response.begin
      yield* request.observe({ progress: { phase: "generating" }, timings: timings() })
      yield* request.finish
      yield* response.end(true)
      yield* tracker.settleRun
    })))
  })
})
