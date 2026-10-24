import { renderToStaticMarkup } from "react-dom/server"
import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ChatGeneration,
  ChatPhase,
  defaultChatSettings,
} from "@magnitudedev/sdk"
import { ChatActivityStatus } from "./chat/activity"
import { ChatResponse } from "./chat/response"
const generation = (patch: Record<string, unknown> = {}) =>
  Schema.decodeUnknownSync(ChatGeneration)({
    id: "gen",
    commandId: "cmd",
    messageId: "msg",
    modelId: "model",
    settings: Schema.encodeSync(typeofSettings)(defaultChatSettings),
    status: "completed",
    createdAt: 100,
    updatedAt: 1100,
    content: "**Answer**\n\n- One\n- Two\n\n```js\nconst answer = 42\n```",
    reasoning: "",
    metrics: {
      outputTokens: 20,
      tokensPerSecond: 40,
      ttftMs: 250,
      inputTokens: 60,
      cachedTokens: 40,
      totalMs: 1000,
    },
    ...patch,
  })
import { ChatSettings as typeofSettings } from "@magnitudedev/sdk"
describe("Chat presentation", () => {
  it("renders Markdown, real inline metrics and response actions without an assistant model heading", () => {
    const html = renderToStaticMarkup(
      <ChatResponse
        generation={generation()}
        retry={() => {}}
        notify={() => {}}
      />
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
        generation={generation({
          content:
            "<script>alert(1)</script>\n\n[Click](javascript:alert(1))\n\n![Tracking](https://example.com/pixel.png)",
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
    for (const [status, label] of [
      ["length", "Output limit reached"],
      ["stopped", "Stopped"],
    ]) {
      const html = renderToStaticMarkup(
        <ChatResponse
          generation={generation({
            status,
            reasoning: "Consider carefully",
            metrics: { reasoningMs: 1000 },
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
  it.each(ChatPhase.literals)(
    "renders %s above the composer with only measured fields",
    (phase) => {
      const g = generation({
        status: "running",
        activity: { phase, since: 100 },
      })
      const html = renderToStaticMarkup(
        <ChatActivityStatus generation={g} now={1100} />
      )
      expect(html).toContain('role="status"')
      expect(html).toContain("Elapsed")
      expect(html).toContain("1.0s")
      expect(html).not.toContain('role="progressbar"')
      expect(html).not.toContain("Remaining")
      expect(html).not.toContain("GB/s")
    }
  )
  it("renders measured prefill counts and compact percentage without a long progress bar", () => {
    const html = renderToStaticMarkup(
      <ChatActivityStatus
        generation={generation({
          status: "running",
          activity: {
            phase: "prefill",
            since: 100,
            fraction: 0.25,
            completed: 2048,
            total: 8192,
            cached: 1536,
          },
        })}
        now={1100}
      />
    )
    expect(html).toContain('aria-valuenow="25"')
    expect(html).toContain("2,048 / 8,192")
    expect(html).toContain("1,536")
    expect(html).not.toContain('style="width:')
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
  for (const status of ["failed", "interrupted"]) {
    const html = renderToStaticMarkup(
      <ChatResponse
        generation={generation({ status, content: "", metrics: {} })}
        notify={() => {}}
      />
    )
    expect(html).toContain(status === "failed" ? "Failed" : "Interrupted")
    expect(html).not.toContain("tok/s")
  }
})
