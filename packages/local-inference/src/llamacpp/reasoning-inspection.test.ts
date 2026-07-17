import { describe, expect, it } from "vitest"
import { Effect, Option } from "effect"
import { discoverLlamaCppReasoning } from "./reasoning-inspection"
import type { LlamaApplyTemplateRequest } from "./server"

const discover = (render: (request: LlamaApplyTemplateRequest) => string) => Effect.runPromise(
  discoverLlamaCppReasoning({ render: (request) => Effect.succeed({ prompt: render(request) }) }),
)

const base = (request: LlamaApplyTemplateRequest) => JSON.stringify({
  messages: request.messages,
  tools: request.tools ?? null,
  toolChoice: request.toolChoice ?? null,
})

describe("llama.cpp differential reasoning inspection", () => {
  it("resolves an effective boolean toggle to none and high", async () => {
    const inspection = await discover((request) => `${base(request)}:${String(request.chatTemplateKwargs?.enable_thinking ?? true)}`)

    expect(inspection.profile.defaultReasoningEffort).toBe("high")
    expect(inspection.profile.effortMappings).toEqual([
      {
        reasoningEffort: "none",
        templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.none() },
        thinkingBudget: { _tag: "Disabled" },
      },
      {
        reasoningEffort: "high",
        templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.none() },
        thinkingBudget: { _tag: "Enabled", tokens: 4_096 },
      },
    ])
  })

  it("resolves ignored controls to none without sending options", async () => {
    const inspection = await discover(base)

    expect(inspection.profile).toEqual({
      defaultReasoningEffort: "none",
      effortMappings: [{
        reasoningEffort: "none",
        templateOptions: { enableThinking: Option.none(), reasoningEffort: Option.none() },
        thinkingBudget: { _tag: "Disabled" },
      }],
    })
  })

  it("maps a validated symbolic high effort to its native option and budget", async () => {
    const inspection = await discover((request) => {
      const effort = request.chatTemplateKwargs?.reasoning_effort
      return `${base(request)}:${effort === "high" ? "high" : "default"}`
    })

    expect(inspection.profile).toEqual({
      defaultReasoningEffort: "high",
      effortMappings: [{
        reasoningEffort: "high",
        templateOptions: { enableThinking: Option.none(), reasoningEffort: Option.some("high") },
        thinkingBudget: { _tag: "Enabled", tokens: 4_096 },
      }],
    })
  })

  it("does not infer efforts from an open pass-through template", async () => {
    const inspection = await discover((request) => `${base(request)}:${String(request.chatTemplateKwargs?.reasoning_effort ?? "default")}`)

    expect(inspection.profile.effortMappings.map((mapping) => mapping.reasoningEffort)).toEqual(["none"])
  })

  it("applies the product budget heuristic to a validated symbolic domain", async () => {
    const inspection = await discover((request) => {
      const effort = String(request.chatTemplateKwargs?.reasoning_effort ?? "default")
      const normalized = ["none", "off", "no_think"].includes(effort)
        ? "none"
        : ["extra_high", "extra-high", "xhigh", "very_high"].includes(effort)
          ? "extra_high"
          : ["minimal", "low", "medium", "high", "max"].includes(effort)
            ? effort
            : "default"
      return `${base(request)}:${normalized}`
    })

    expect(inspection.profile.defaultReasoningEffort).toBe("high")
    expect(inspection.profile.effortMappings.map((mapping) => [
      mapping.reasoningEffort,
      mapping.thinkingBudget,
    ])).toEqual([
      ["none", { _tag: "Disabled" }],
      ["minimal", { _tag: "Enabled", tokens: 1_024 }],
      ["low", { _tag: "Enabled", tokens: 1_024 }],
      ["medium", { _tag: "Enabled", tokens: 2_048 }],
      ["high", { _tag: "Enabled", tokens: 4_096 }],
      ["extra_high", { _tag: "Enabled", tokens: 8_192 }],
      ["max", { _tag: "Disabled" }],
    ])
  })

  it("combines an effective toggle with validated native efforts", async () => {
    const inspection = await discover((request) => {
      const enabled = request.chatTemplateKwargs?.enable_thinking ?? true
      const effort = request.chatTemplateKwargs?.reasoning_effort
      const branch = enabled === false
        ? "none"
        : effort === "low" || effort === "high"
          ? effort
          : "enabled"
      return `${base(request)}:${branch}`
    })

    expect(inspection.profile.effortMappings).toEqual([
      {
        reasoningEffort: "none",
        templateOptions: { enableThinking: Option.some(false), reasoningEffort: Option.none() },
        thinkingBudget: { _tag: "Disabled" },
      },
      {
        reasoningEffort: "low",
        templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.some("low") },
        thinkingBudget: { _tag: "Enabled", tokens: 1_024 },
      },
      {
        reasoningEffort: "high",
        templateOptions: { enableThinking: Option.some(true), reasoningEffort: Option.some("high") },
        thinkingBudget: { _tag: "Enabled", tokens: 4_096 },
      },
    ])
  })
})
