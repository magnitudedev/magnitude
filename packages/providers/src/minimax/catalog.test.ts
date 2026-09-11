import { FetchHttpClient } from "@effect/platform"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { createMiniMaxCatalog, MINIMAX_PROVIDER_ID } from "./catalog"

describe("MiniMax model catalog", () => {
  it("publishes the current text models with authoritative capabilities", async () => {
    const models = await Effect.runPromise(
      createMiniMaxCatalog().list.pipe(Effect.provide(FetchHttpClient.layer)),
    )

    expect(models).toHaveLength(2)
    expect(models[0]).toMatchObject({
      providerId: MINIMAX_PROVIDER_ID,
      providerModelId: "MiniMax-M3",
      contextWindow: 1_000_000,
      maxOutputTokens: 524_288,
      defaultReasoningEffort: "adaptive",
      servingCapabilities: { tools: true, structuredOutput: false },
    })
    expect(models[0]?.properties).toMatchObject({
      vision: { _tag: "Resolved", value: true },
      reasoning: { _tag: "Resolved", value: ["adaptive", "disabled"] },
    })
    expect(Option.getOrThrow(models[0]!.pricing)).toEqual({
      input: 0.6,
      output: 2.4,
      cached_input: 0.12,
    })
    expect(models[1]).toMatchObject({
      providerModelId: "MiniMax-M2.7",
      contextWindow: 204_800,
      maxOutputTokens: 204_800,
      defaultReasoningEffort: "always_on",
    })
    expect(models[1]?.properties).toMatchObject({
      vision: { _tag: "Resolved", value: false },
      reasoning: { _tag: "Resolved", value: ["always_on"] },
    })
  })
})
