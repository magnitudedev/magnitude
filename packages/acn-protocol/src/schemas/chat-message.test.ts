import { describe, expect, it } from "vitest"
import { DateTime, Option, Schema } from "effect"
import {
  appendOutput,
  ChatAssistantMessage,
  type ChatContentPart,
  ChatMessage,
  decodeRate,
  responseMetrics,
  responseStatus,
  responseStatusText,
  runningResponse,
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
      tokensPerSecond: Option.some(38),
      // The 20 prompt tokens not found in the cache, over 50 ms.
      prefillTokensPerSecond: Option.some(400),
      inputTokens: Option.some(60),
      cachedTokens: Option.some(40),
      outputTokens: Option.some(20),
    })
  })

  it("gives no speed for a response of one token, which decoded nothing after the prompt", () => {
    const one = responseMetrics(ended({ type: "failed", error: { type: "inference", code: "worker_exited", message: "x" } },
      { usage: { ...usage, completionTokens: 1 }, timings: { ...timings, predictedMs: 0.02 } }))
    expect(one.tokensPerSecond).toEqual(Option.none())
    expect(one.outputTokens).toEqual(Option.some(1))
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
      Option.some("This chat is longer than the context this computer serves this model at. Start a new chat to continue."),
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
    expect(responseStatusText(message)).toMatchObject({ label, problem })
  })
})

describe("Response failure text", () => {
  it.each(["unreachable", "rejected", "storage", "protocol"])("shows a %s failure's own message", (type) => {
    expect(responseStatusText(ended({ type: "failed", error: { type, message: "Went wrong." } })))
      .toEqual({ label: Option.some("Failed"), problem: Option.some("Went wrong."), detail: Option.none() })
  })

  const backtrace = `worker lost during load: the worker ended before it was ready; worker signal: 6 (SIGABRT) (core dumped); worker diagnostics: ${
    Array.from({ length: 62 }, (_, frame) => `${frame}: 0x5a620e5e9763 - std::sys::backtrace::frame at /home/ubuntu/inference/engine/scheduler/src/worker.rs:429:28`).join(" ")}`
  const inference = (code: string, message = backtrace) => ended({ type: "failed", error: { type: "inference", code, message } })

  it.each([
    ["worker_exited", "The model’s engine stopped unexpectedly. Try again, or choose another model."],
    ["worker_unresponsive", "The model’s engine stopped responding. Try again, or choose another model."],
    ["model_load_failed", "The model couldn’t be loaded. Try again, or choose another model."],
    ["model_not_ready", "The model isn’t ready yet. Wait for it to finish loading, then try again."],
    ["insufficient_memory", "There wasn’t enough free memory for this model. Close other apps or stop another model, then try again."],
    ["context_exceeds_memory", "This chat needs more memory than this computer can give the model. Start a new chat, or choose a smaller model."],
    ["context_length_exceeded", "This chat is longer than the context this computer serves this model at. Start a new chat to continue."],
    ["device_lost", "The device running the model stopped working. Try again; if it keeps happening, restart Magnitude."],
    ["backend_error", "The inference engine hit an error. Try again, or choose another model."],
    ["something_new", "The inference engine reported an error (something_new). Try again, or choose another model."],
  ])("says what a %s failure means in its own sentence, keeping the service's report as its detail", (code, sentence) => {
    const { problem, detail } = responseStatusText(inference(code))
    expect(problem).toEqual(Option.some(sentence))
    expect(detail).toEqual(Option.some(backtrace))
  })

  it("never puts the service's diagnostic text in the summary", () => {
    const problem = Option.getOrThrow(responseStatusText(inference("worker_exited")).problem)
    expect(problem).not.toContain("worker diagnostics")
    expect(problem).not.toContain("backtrace")
    expect(problem.length).toBeLessThan(120)
  })

  it("claims no phase a code doesn't carry: a worker exits while loading as well as while answering", () => {
    for (const code of ["worker_exited", "worker_unresponsive", "model_unloaded", "model_instance_stopped", "device_lost"]) {
      const problem = Option.getOrThrow(responseStatusText(inference(code, "worker lost during load")).problem)
      expect(problem).not.toMatch(/answering|loading|while/i)
    }
  })

  it("has no detail when the service sent no message", () => {
    expect(responseStatusText(inference("worker_exited", " ")).detail).toEqual(Option.none())
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

describe("runningResponse", () => {
  const user = Schema.decodeUnknownSync(ChatMessage)({ type: "user", id: "message", text: "Hi", time: 900 })

  it("is the latest response while it has no end: from its start, through loading, thinking and streaming, until a stop ends it", () => {
    const started = response({ end: undefined, content: [] })
    expect(runningResponse([user, started])).toEqual(Option.some(started))
    const streaming = response({ end: undefined })
    expect(runningResponse([user, streaming])).toEqual(Option.some(streaming))
  })

  it("is none once the latest response ended, however it ended, and none before any response", () => {
    for (const outcome of [
      { type: "finished", finishReason: "stop" },
      { type: "failed", error: { type: "inference", code: "worker_exited", message: "x" } },
      { type: "stopped", by: "user" },
      { type: "stopped", by: "shutdown" },
    ]) expect(runningResponse([user, ended(outcome)])).toEqual(Option.none())
    expect(runningResponse([user])).toEqual(Option.none())
  })

  it("follows only the latest response, not an earlier one left without an end", () => {
    expect(runningResponse([response({ end: undefined, id: "earlier" }), user, ended({ type: "finished", finishReason: "stop" })])).toEqual(Option.none())
  })
})

describe("decode rate", () => {
  it("counts each response's tokens after its first, and none where decode made nothing", () => {
    expect(decodeRate(7, 3)).toEqual(Option.some(2_000))
    expect(decodeRate(32, 1_500, 2)).toEqual(Option.some(20))
    expect(decodeRate(1, 0.02)).toEqual(Option.none())
    expect(decodeRate(5, 0)).toEqual(Option.none())
  })
})
