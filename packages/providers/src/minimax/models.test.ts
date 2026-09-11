import { FetchHttpClient } from "@effect/platform"
import { Auth, PromptBuilder } from "@magnitudedev/ai"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { createMiniMaxCompatibleSpec } from "./models"

describe("MiniMax Chat Completions models", () => {
  it.each([
    {
      modelId: "MiniMax-M3",
      reasoningMode: "configurable" as const,
      reasoningEffort: "disabled",
      expectedThinking: { type: "disabled" },
    },
    {
      modelId: "MiniMax-M2.7",
      reasoningMode: "always_on" as const,
      reasoningEffort: "always_on",
      expectedThinking: undefined,
    },
  ])("encodes $modelId request options", async ({
    modelId,
    reasoningMode,
    reasoningEffort,
    expectedThinking,
  }) => {
    let requestUrl = ""
    let authorization = ""
    let requestBody: Record<string, unknown> = {}
    const server = Bun.serve({
      port: 0,
      fetch: async (request) => {
        requestUrl = request.url
        authorization = request.headers.get("Authorization") ?? ""
        requestBody = await request.json() as Record<string, unknown>
        return new Response("data: [DONE]\n\n", {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })

    try {
      const spec = createMiniMaxCompatibleSpec({
        modelId,
        endpoint: `http://127.0.0.1:${server.port}/v1`,
        reasoningMode,
      })
      await Effect.runPromise(spec.bind({ auth: Auth.bearer("test-key") }).stream(
        PromptBuilder.empty().user("hello").build(),
        [],
        { maxTokens: 128, reasoningEffort },
      ).pipe(Effect.provide(FetchHttpClient.layer)))

      expect(requestUrl).toBe(`http://127.0.0.1:${server.port}/v1/chat/completions`)
      expect(authorization).toBe("Bearer test-key")
      expect(requestBody).toMatchObject({
        model: modelId,
        max_completion_tokens: 128,
        reasoning_split: true,
      })
      if (expectedThinking === undefined) expect(requestBody).not.toHaveProperty("thinking")
      else expect(requestBody).toHaveProperty("thinking", expectedThinking)
    } finally {
      server.stop(true)
    }
  })
})
