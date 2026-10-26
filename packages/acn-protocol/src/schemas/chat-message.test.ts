import { describe, expect, it } from "vitest"
import { DateTime, Option, Schema } from "effect"
import {
  appendOutput,
  ChatAssistantMessage,
  type ChatContentPart,
  responseMetrics,
  responseStatus,
  responseStatusText,
} from "./chat-message"

const usage = { promptTokens: 60, cachedTokens: 40, completionTokens: 20 }
const timings = { promptMs: 50, predictedMs: 500, samplerMs: 1, parserMs: 1, timeToFirstTokenMs: 240 }
const encoded = {
  type: "assistant",
  id: "response",
  model: "qwen3-8b",
  settings: { systemPrompt: "", stop: [] },
  started: 1_000,
  content: [
    { type: "reasoning", text: "Hm", time: { started: 1_250, ended: 1_650 } },
    { type: "text", text: "Hi", time: { started: 1_700, ended: 1_900 } },
    { type: "reasoning", text: "Again", time: { started: 1_950, ended: 2_050 } },
  ],
  end: {
    time: 2_100,
    outcome: { type: "finished", finishReason: "stop" },
    usage,
    timings,
  },
}
// A response with these fields changed; an `undefined` field is left out.
const response = (patch: Record<string, unknown> = {}) =>
  Schema.decodeUnknownSync(ChatAssistantMessage)(
    Object.fromEntries(Object.entries({ ...encoded, ...patch }).filter(([, value]) => value !== undefined))
  )
const ended = (outcome: unknown, rest: Record<string, unknown> = {}) => response({ end: { time: 2_100, outcome, ...rest } })
const part = (tag: "text" | "reasoning", text: string, started: number, endedAt: number): ChatContentPart => ({
  _tag: tag,
  text,
  time: { started: DateTime.unsafeMake(started), ended: DateTime.unsafeMake(endedAt) },
})

describe("Response metrics", () => {
  it("derives every measurement from the transcript's times, timings and usage", () => {
    expect(responseMetrics(response())).toEqual({
      ttftMs: Option.some(250),
      firstAnswerMs: Option.some(700),
      thoughtForMs: Option.some(500),
      totalMs: Option.some(1_100),
      generationMs: Option.some(500),
      tokensPerSecond: Option.some(40),
      // The 20 prompt tokens not found in the cache, over 50 ms.
      prefillTokensPerSecond: Option.some(400),
      inputTokens: Option.some(60),
      cachedTokens: Option.some(40),
      outputTokens: Option.some(20),
    })
  })

  it("counts a stopped response's tokens from its report, and none for one stopped before any", () => {
    const stopped = responseMetrics(ended({ type: "stopped", by: "user" }, { usage, timings }))
    expect([stopped.inputTokens, stopped.cachedTokens, stopped.outputTokens]).toEqual([Option.some(60), Option.some(40), Option.some(20)])
    const queued = responseMetrics(ended({ type: "stopped", by: "user" }))
    for (const value of [queued.inputTokens, queued.outputTokens, queued.tokensPerSecond, queued.generationMs]) expect(value).toEqual(Option.none())
  })

  it("leaves out what a response never observed", () => {
    const metrics = responseMetrics(response({ content: [], end: undefined }))
    for (const value of Object.values(metrics)) expect(value).toEqual(Option.none())
  })

  it("reports no rate or generation time for a span that measured no time", () => {
    const metrics = responseMetrics(ended(
      { type: "finished", finishReason: "stop" },
      { usage, timings: { ...timings, promptMs: 0, predictedMs: 0 } }
    ))
    expect(metrics.tokensPerSecond).toEqual(Option.none())
    expect(metrics.prefillTokensPerSecond).toEqual(Option.none())
    expect(metrics.generationMs).toEqual(Option.none())
  })

  it("has no thinking time for a response that only answered", () => {
    const metrics = responseMetrics(response({
      content: [{ type: "text", text: "Hi", time: { started: 1_300, ended: 1_400 } }],
    }))
    expect(metrics.ttftMs).toEqual(Option.some(300))
    expect(metrics.firstAnswerMs).toEqual(Option.some(300))
    expect(metrics.thoughtForMs).toEqual(Option.none())
  })
})

describe("Response status", () => {
  it.each([
    [undefined, "running", Option.none(), Option.none()],
    [{ type: "finished", finishReason: "stop" }, "completed", Option.none(), Option.none()],
    [{ type: "finished", finishReason: "length" }, "length", Option.some("Output limit reached"), Option.none()],
    [
      { type: "finished", finishReason: "tool_calls" },
      "failed",
      Option.some("Failed"),
      Option.some("The model ended with tool_calls. This chat has no tools."),
    ],
    [
      { type: "failed", error: { type: "inference", code: "context_length_exceeded", message: "Prompt too long" } },
      "failed",
      Option.some("Failed"),
      Option.some("The inference service failed: Prompt too long"),
    ],
    [{ type: "stopped", by: "user" }, "stopped", Option.some("Stopped"), Option.none()],
    [
      { type: "stopped", by: "shutdown" },
      "interrupted",
      Option.some("Interrupted"),
      Option.some("The response ended before completion. Retry to continue."),
    ],
  ])("derives %j as %s", (outcome, status, label, problem) => {
    const message = outcome === undefined ? response({ end: undefined }) : ended(outcome)
    expect(responseStatus(message)).toBe(status)
    expect(responseStatusText(message)).toEqual({ label, problem })
  })
})

describe("Response failure text", () => {
  it.each(["unreachable", "rejected", "storage", "protocol"])("shows a %s failure's own message", (type) => {
    expect(responseStatusText(ended({ type: "failed", error: { type, message: "Went wrong." } })))
      .toEqual({ label: Option.some("Failed"), problem: Option.some("Went wrong.") })
  })
})

describe("appendOutput", () => {
  it("joins consecutive output of one part, from its first time to its last", () => {
    const content = appendOutput([part("reasoning", "Hm", 0, 5)], part("reasoning", "m", 9, 12))
    expect(content).toEqual([part("reasoning", "Hmm", 0, 12)])
  })

  it("opens a new part when the part changes", () => {
    expect(appendOutput([part("reasoning", "Hm", 0, 5)], part("text", "Hi", 7, 8)))
      .toEqual([part("reasoning", "Hm", 0, 5), part("text", "Hi", 7, 8)])
  })
})
