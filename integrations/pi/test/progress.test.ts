import { describe, expect, it, vi } from "vitest"
import { MAGNITUDE_SUMMARY_WIDGET_KEY, MagnitudeProgressTracker } from "../extensions/progress"

const expectLastSummary = (setWidget: ReturnType<typeof vi.fn>, summary: string): void => {
  const factory = setWidget.mock.calls.at(-1)?.[1] as
    | ((tui: unknown, theme: { fg: (color: string, text: string) => string }) => { render: (width: number) => string[] })
    | undefined
  expect(factory).toBeTypeOf("function")
  const fg = vi.fn((_color: string, text: string) => text)
  const component = factory?.({}, { fg })
  expect(fg).toHaveBeenCalledWith("muted", summary)
  expect(component?.render(200).map((line) => line.trimEnd())).toEqual([summary])
}

describe("MagnitudeProgressTracker", () => {
  it("renders the approved live stages and derives uncached prefill counts", () => {
    let now = 1_000
    let tick = () => {}
    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    const tracker = new MagnitudeProgressTracker({ setWidget, setWorkingMessage }, {
      now: () => now,
      setInterval: (callback) => {
        tick = callback
        return {} as ReturnType<typeof setInterval>
      },
      clearInterval: vi.fn(),
    })
    const request = tracker.begin("Qwen3.6 35B-A3B (Q6)")

    request.observe({ progress: { phase: "model_loading", fraction: 0.47 } })
    now = 3_300
    tick()
    expect(setWorkingMessage).toHaveBeenLastCalledWith(
      "Loading Qwen3.6 35B-A3B (Q6) into memory · 47% · 2.3s",
    )

    request.observe({ progress: { phase: "preparing" } })
    expect(setWorkingMessage).toHaveBeenLastCalledWith()

    request.observe({
      progress: {
        phase: "prefill",
        completed_tokens: 14_020,
        total_tokens: 14_300,
        cached_tokens: 13_200,
      },
    })
    now = 3_700
    tick()
    expect(setWorkingMessage).toHaveBeenLastCalledWith(
      "Prefilling prompt · 820 / 1.1k tokens · 13.2k cached · 0.4s",
    )

    request.observe({ progress: { phase: "generating" } })
    now = 5_400
    tick()
    expect(setWorkingMessage).toHaveBeenLastCalledWith("Working · 1.7s")
    expect(setWidget).toHaveBeenLastCalledWith(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
  })

  it("summarizes a settled multi-request run with first TTFT and weighted throughput", () => {
    let now = 0
    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    const tracker = new MagnitudeProgressTracker({ setWidget, setWorkingMessage }, {
      now: () => now,
      setInterval: () => ({} as ReturnType<typeof setInterval>),
      clearInterval: vi.fn(),
    })
    tracker.startRun("Qwen3.6 35B-A3B (Q6)")
    const first = tracker.begin("Qwen3.6 35B-A3B (Q6)")
    first.observe({ timings: {
      prompt_ms: 500,
      time_to_first_token_ms: 3_400,
      predicted_n: 10,
      predicted_ms: 200,
      predicted_per_second: 50,
    } })
    first.finish()

    now = 4_000
    const second = tracker.begin("Qwen3.6 35B-A3B (Q6)")
    second.observe({ timings: {
      prompt_ms: 100,
      time_to_first_token_ms: 500,
      predicted_n: 20,
      predicted_ms: 200,
      predicted_per_second: 100,
    } })
    now = 6_000
    tracker.settleRun()
    expect(setWidget).not.toHaveBeenCalledWith(
      MAGNITUDE_SUMMARY_WIDGET_KEY,
      expect.arrayContaining([expect.stringContaining("worked for")]),
    )
    second.finish()
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    expectLastSummary(
      setWidget,
      "● Qwen3.6 35B-A3B (Q6) worked for 6 seconds · 3.4s TTFT · 75.0 tok/s",
    )

    tracker.startRun("Qwen3.6 35B-A3B (Q6)")
    expect(setWidget).toHaveBeenLastCalledWith(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
  })

  it("uses Magnitude's completed-work duration formatting", () => {
    let now = 0
    const setWidget = vi.fn()
    const tracker = new MagnitudeProgressTracker({ setWidget, setWorkingMessage: vi.fn() }, {
      now: () => now,
      setInterval: () => ({} as ReturnType<typeof setInterval>),
      clearInterval: vi.fn(),
    })
    const completeRun = (settledAt: number) => {
      tracker.startRun("Model")
      const request = tracker.begin("Model")
      request.observe({ timings: {
        prompt_ms: 1,
        time_to_first_token_ms: 100,
        predicted_n: 1,
        predicted_ms: 10,
        predicted_per_second: 100,
      } })
      request.finish()
      now = settledAt
      tracker.settleRun()
    }

    completeRun(999)
    expectLastSummary(
      setWidget,
      "● Model worked for <1 second · 0.1s TTFT · 100.0 tok/s",
    )

    now = 10_000
    completeRun(75_000)
    expectLastSummary(
      setWidget,
      "● Model worked for 1:05 · 0.1s TTFT · 100.0 tok/s",
    )
  })

  it("clamps inconsistent server counters before deriving uncached prefill", () => {
    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    const tracker = new MagnitudeProgressTracker({ setWidget, setWorkingMessage }, {
      now: () => 0,
      setInterval: () => ({} as ReturnType<typeof setInterval>),
      clearInterval: vi.fn(),
    })
    const request = tracker.begin("Qwen3.6 35B-A3B (Q6)")

    request.observe({
      progress: {
        phase: "prefill",
        completed_tokens: 9_000,
        total_tokens: 1_000,
        cached_tokens: 2_000,
      },
    })

    expect(setWorkingMessage).toHaveBeenLastCalledWith(
      "Prefilling prompt · 0 / 0 tokens · 1k cached · 0.0s",
    )
  })

  it("clears incomplete, failed, model-switched, and disposed state", () => {
    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    const clearInterval = vi.fn()
    const tracker = new MagnitudeProgressTracker({ setWidget, setWorkingMessage }, {
      setInterval: () => ({ id: 7 } as unknown as ReturnType<typeof setInterval>),
      clearInterval,
    })
    const incomplete = tracker.begin("Qwen3.6 35B-A3B (Q6)")
    incomplete.observe({ progress: { phase: "generating" } })
    incomplete.finish()
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    expect(setWidget).toHaveBeenLastCalledWith(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
    const failed = tracker.begin("Qwen3.6 35B-A3B (Q6)")
    failed.observe({ progress: { phase: "generating" } })
    failed.fail()
    tracker.settleRun()
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    expect(setWidget).toHaveBeenLastCalledWith(MAGNITUDE_SUMMARY_WIDGET_KEY, undefined)
    tracker.clear()
    tracker.dispose()
    tracker.dispose()
    expect(clearInterval).toHaveBeenCalledTimes(2)
    expect(clearInterval).toHaveBeenNthCalledWith(1, { id: 7 })
    expect(clearInterval).toHaveBeenNthCalledWith(2, { id: 7 })
  })
})
