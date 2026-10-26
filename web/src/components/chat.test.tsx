import { renderToStaticMarkup } from "react-dom/server"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ChatAssistantMessage, LiveResponse } from "@magnitudedev/sdk"
import { ChatActivityStatus } from "./chat/activity"
import { ChatResponse } from "./chat/response"

const answer = "**Answer**\n\n- One\n- Two\n\n```js\nconst answer = 42\n```"
const encoded = {
  type: "assistant",
  id: "response",
  model: "model",
  settings: { systemPrompt: "", stop: [] },
  started: 100,
  content: [{ type: "text", text: answer, time: { started: 350, ended: 1_000 } }],
  end: {
    time: 1_100,
    outcome: { type: "finished", finishReason: "stop" },
    usage: { promptTokens: 60, cachedTokens: 40, completionTokens: 20 },
    timings: { promptMs: 50, predictedMs: 500, samplerMs: 1, parserMs: 1, timeToFirstTokenMs: 240 },
  },
}
const response = (patch: Record<string, unknown> = {}) =>
  Schema.decodeUnknownSync(ChatAssistantMessage)({ ...encoded, ...patch })
/** A response that ended this way, with no measurements. */
const ended = (outcome: unknown, patch: Record<string, unknown> = {}) =>
  response({ content: [], end: { time: 1_100, outcome }, ...patch })
const running = (patch: Record<string, unknown> = {}) => {
  const { end: _, ...unended } = encoded
  return Schema.decodeUnknownSync(ChatAssistantMessage)({ ...unended, content: [], ...patch })
}
const live = (encodedPhase: unknown, progress?: unknown, report?: { readonly usage: unknown; readonly timings: unknown }) =>
  Option.some(Schema.decodeUnknownSync(LiveResponse)({
    id: "response",
    phase: { phase: encodedPhase, ...(progress === undefined ? {} : { progress }) },
    ...report,
  }))
const tuning = (operation?: string) => ({
  type: "model_loading",
  details: {
    kind: "tuning", measured_configurations: 155, formed_programs: 142, elapsed_ms: 18_900, budget_ms: 60_000,
    ...(operation ? { work: { entry: "attention_decode", operation } } : {}),
  },
})

describe("Chat presentation", () => {
  it("renders Markdown, real inline metrics and response actions without an assistant model heading", () => {
    const html = renderToStaticMarkup(
      <ChatResponse response={response()} retry={() => {}} notify={() => {}} />
    )
    expect(html).toContain("<strong>Answer</strong>")
    expect(html).toContain("<ul>")
    expect(html).toContain('aria-label="JavaScript code"')
    expect(html).toContain('aria-label="Copy code"')
    expect(html).toContain('class="hljs-keyword"')
    expect(html).toContain("40 tok/s")
    expect(html).toContain("TTFT 0.25s")
    expect(html).toContain("20 tokens")
    expect(html).toContain("Retry response")
    expect(html).not.toContain("model logo")
    expect(html).not.toContain("Response details")
  })
  it("does not execute model HTML, unsafe links or remote images", () => {
    const html = renderToStaticMarkup(
      <ChatResponse
        response={response({
          content: [{
            type: "text",
            text: "<script>alert(1)</script>\n\n[Click](javascript:alert(1))\n\n![Tracking](https://example.com/pixel.png)",
            time: { started: 350, ended: 1_000 },
          }],
        })}
        notify={() => {}}
      />
    )
    expect(html).not.toContain("<script>")
    expect(html).not.toContain('href="javascript:')
    expect(html).not.toContain("<img")
    expect(html).not.toContain('src="https://example.com')
  })
  it("keeps reasoning separate and distinguishes an output limit from Stop", () => {
    for (const [outcome, label] of [
      [{ type: "finished", finishReason: "length" }, "Output limit reached"],
      [{ type: "stopped", by: "user" }, "Stopped"],
    ] as const) {
      const html = renderToStaticMarkup(
        <ChatResponse
          response={ended(outcome, {
            content: [{ type: "reasoning", text: "Consider carefully", time: { started: 200, ended: 1_200 } }],
          })}
          notify={() => {}}
        />
      )
      expect(html).toContain("<details")
      expect(html).toContain("Thought for 1.0s")
      expect(html).toContain(label)
      expect(html).not.toContain("tok/s")
    }
  })
  it("shows Thinking while a running response has only reasoned", () => {
    const html = renderToStaticMarkup(
      <ChatResponse
        response={running({ content: [{ type: "reasoning", text: "Hm", time: { started: 200, ended: 300 } }] })}
        notify={() => {}}
      />
    )
    expect(html).toContain(">Thinking</summary>")
  })

  const statuses = [
    ["starting", Option.none(), "Starting request"],
    ["queued", live({ type: "queued" }), "Waiting for inference"],
    ["preparing", live({ type: "preparing" }), "Preparing prompt"],
    ["model queued", live({ type: "model_loading", stage: "queued" }), "Waiting for model"],
    ["model preparing", live({ type: "model_loading", stage: "preparing" }), "Preparing model"],
    ["optimizing", live({ type: "model_loading", stage: "optimizing" }), "Optimizing model"],
    ["loading weights", live({ type: "model_loading", stage: "loading_weights" }), "Loading model"],
    ["finalizing", live({ type: "model_loading", stage: "finalizing" }), "Finalizing model"],
    ["prefill", live({ type: "prefill" }), "Processing prompt"],
  ] as const
  it.each(statuses)("renders the %s status with the request's elapsed time", (_, current, title) => {
    const html = renderToStaticMarkup(<ChatActivityStatus response={running()} live={current} now={1100} />)
    expect(html).toContain('role="status"')
    expect(html).toContain(`aria-label="${title}"`)
    expect(html).toContain('aria-label="Request elapsed time"')
    expect(html).toContain("00:01")
    expect(html).not.toContain('role="progressbar"')
    expect(html).not.toContain("aria-expanded")
    expect(html).not.toContain("·")
  })
  it("renders Thinking with its live output rate while reasoning is generated, and nothing while the answer is", () => {
    const generating = live({ type: "generating" }, undefined, {
      usage: { promptTokens: 20, cachedTokens: 0, completionTokens: 1_234 },
      timings: { promptMs: 50, predictedMs: 30_000, samplerMs: 1, parserMs: 1, timeToFirstTokenMs: 240 },
    })
    const reasoning = running({ content: [{ type: "reasoning", text: "Hm", time: { started: 200, ended: 300 } }] })
    const thinking = renderToStaticMarkup(<ChatActivityStatus response={reasoning} live={generating} now={1100} />)
    expect(thinking).toContain('aria-label="Thinking"')
    expect(thinking).toContain("1,234 tokens")
    expect(thinking).toContain("41.1 tok/s")
    const answering = running({ content: [{ type: "text", text: "Hi", time: { started: 200, ended: 300 } }] })
    expect(renderToStaticMarkup(<ChatActivityStatus response={answering} live={generating} now={1100} />)).toBe("")
    expect(renderToStaticMarkup(<ChatActivityStatus response={running()} live={generating} now={1100} />)).toBe("")
  })
  it("renders the tested kernel count while optimizing, without tuning internals", () => {
    const html = renderToStaticMarkup(
      <ChatActivityStatus response={running()} live={live({ type: "model_loading", stage: "optimizing" }, tuning("measure"))} now={1100} />
    )
    expect(html).toContain("Testing kernel variants")
    expect(html).toContain("155 tested")
    expect(html).not.toContain("programs")
    expect(html).not.toContain("budget")
    expect(html).not.toContain("attention_decode")
    expect(html).not.toContain('role="progressbar"')
  })
  it("keeps optimization wording stable across rapid tuning operations and missing work details", () => {
    const render = (operation?: string) => renderToStaticMarkup(
      <ChatActivityStatus response={running()} live={live({ type: "model_loading", stage: "optimizing" }, tuning(operation))} now={1100} />
    )
    for (const operation of ["discovery", "form", "measure", "promote", "confirm"]) {
      expect(render(operation)).toBe(render())
    }
  })
  it("renders measured prefill counts and compact percentage without a long progress bar", () => {
    const html = renderToStaticMarkup(
      <ChatActivityStatus
        response={running()}
        live={live({ type: "prefill" }, { type: "prefill", completedTokens: 2048, totalTokens: 8192, cachedTokens: 1536 })}
        now={1100}
      />
    )
    expect(html).toContain('aria-valuenow="25"')
    expect(html).toContain("2,048 / 8,192")
    expect(html).toContain("1,536 cached")
    expect(html).not.toContain("aria-expanded")
    expect(html).not.toContain('style="width:')
  })
  it("renders the weight bytes loaded and a compact percentage", () => {
    const html = renderToStaticMarkup(
      <ChatActivityStatus
        response={running()}
        live={live(
          { type: "model_loading", stage: "loading_weights" },
          { type: "model_loading", details: { kind: "weights", completed_bytes: 2 * 1024 ** 3, total_bytes: 8 * 1024 ** 3 } }
        )}
        now={1100}
      />
    )
    expect(html).toContain('aria-valuenow="25"')
    expect(html).toContain("2.0 / 8.0 GB")
  })
})

import { submitsChatMessage } from "./chat/keyboard"
it("keeps IME composition and Shift-Enter out of message submission", () => {
  expect(submitsChatMessage("Enter", false, false, 13)).toBe(true)
  expect(submitsChatMessage("Enter", true, false, 13)).toBe(false)
  expect(submitsChatMessage("Enter", false, true, 13)).toBe(false)
  expect(submitsChatMessage("Enter", false, false, 229)).toBe(false)
  expect(submitsChatMessage("a", false, false, 65)).toBe(false)
})

import { formatStopSequences, parseStopSequences } from "./chat/stop-sequences"
it("edits newline and literal backslash stop sequences without losing their value", () => {
  const sequences = ["\n", "END", "\t", String.raw`\n`, String.raw`\d+`]
  expect(parseStopSequences(formatStopSequences(sequences))).toEqual(sequences)
  expect(parseStopSequences("END\n\nSTOP")).toEqual(["END", "STOP"])
})

it("labels responses with no token measurements instead of showing an empty metrics control", () => {
  const failed = (outcome: unknown) =>
    renderToStaticMarkup(<ChatResponse response={ended(outcome)} notify={() => {}} />)
  const reported = failed({ type: "failed", error: { type: "inference", code: "server_error", message: "The engine stopped." } })
  expect(reported).toContain("Failed")
  expect(reported).toContain("The engine stopped.")
  expect(reported).not.toContain("tok/s")
  const unfinished = failed({ type: "finished", finishReason: "tool_calls" })
  expect(unfinished).toContain("The model ended with tool_calls. This chat has no tools.")
})

it("labels a response a shutdown ended as interrupted, and one the user ended as stopped", () => {
  const render = (by: string) =>
    renderToStaticMarkup(<ChatResponse response={ended({ type: "stopped", by })} notify={() => {}} />)
  const interrupted = render("shutdown")
  expect(interrupted).toContain("Interrupted")
  expect(interrupted).toContain("The response ended before completion. Retry to continue.")
  const stopped = render("user")
  expect(stopped).toContain("Stopped")
  expect(stopped).not.toContain("Retry to continue")
})

it("shows a stopped response's token count from the engine's report", () => {
  const html = renderToStaticMarkup(
    <ChatResponse
      response={ended({ type: "stopped", by: "user" }, {
        end: { time: 1_100, outcome: { type: "stopped", by: "user" }, usage: encoded.end.usage, timings: encoded.end.timings },
      })}
      notify={() => {}}
    />
  )
  expect(html).toContain("20 tokens")
  expect(html).toContain("Stopped")
})
