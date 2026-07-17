import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import { discoverLlamaCppReasoning } from "./reasoning-discovery"
import type { LlamaApplyTemplateRequest } from "./server"

const discover = (render: (request: LlamaApplyTemplateRequest) => string) => Effect.runPromise(
  discoverLlamaCppReasoning({ render: (request) => Effect.succeed({ prompt: render(request) }) }),
)

const base = (request: LlamaApplyTemplateRequest) => JSON.stringify({
  messages: request.messages,
  tools: request.tools ?? null,
  toolChoice: request.toolChoice ?? null,
})

describe("llama.cpp differential reasoning discovery", () => {
  it("discovers an effective Qwen-style thinking toggle", async () => {
    const inspection = await discover((request) => `${base(request)}:${String(request.chatTemplateKwargs?.enable_thinking ?? true)}`)
    expect(inspection.options.map(({ reasoningEffort }) => reasoningEffort)).toEqual(["Default", "None"])
    expect(inspection.options[1]?.control).toEqual({ _tag: "EnableThinkingKwarg", enabled: false })
  })

  it("retains only Default when a Kimi-style template ignores both controls", async () => {
    const inspection = await discover(base)
    expect(inspection.options).toEqual([{ reasoningEffort: "Default", control: { _tag: "Omitted" } }])
  })

  it("discovers only symbolic values distinct from an invalid-value fallback", async () => {
    const inspection = await discover((request) => {
      const effort = request.chatTemplateKwargs?.reasoning_effort
      return `${base(request)}:${effort === "high" ? "high" : "default"}`
    })
    expect(inspection.options.map(({ reasoningEffort }) => reasoningEffort)).toEqual(["Default", "High"])
  })

  it("does not claim a domain for an open pass-through template", async () => {
    const inspection = await discover((request) => `${base(request)}:${String(request.chatTemplateKwargs?.reasoning_effort ?? "default")}`)
    expect(inspection.options).toEqual([{ reasoningEffort: "Default", control: { _tag: "Omitted" } }])
  })
})
