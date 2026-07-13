import { Effect, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  Auth,
  Model,
  NativeChatCompletions,
  Option,
  PromptBuilder,
} from "../src/index.js"

// Define a model spec for Fireworks AI
const fireworksLlama = NativeChatCompletions.model({
  id: "fireworks/llama-v3p1-8b",
  modelId: "accounts/fireworks/models/llama-v3p1-8b-instruct",
  endpoint: "https://api.fireworks.ai/inference/v1",
  contextWindow: 131_072,
  maxOutputTokens: 16_384,
  options: {
    ...NativeChatCompletions.options,
    temperature: Option.define((v: number) => ({ temperature: v }), 0.7),
  },
})

// Bind with auth
const apiKey = process.env.FIREWORKS_API_KEY
if (!apiKey) throw new Error("Set FIREWORKS_API_KEY before running this example")

const model = fireworksLlama.bind({ auth: Auth.bearer(apiKey) })

// Build prompt
const prompt = PromptBuilder.empty()
  .system("You are a concise, helpful assistant.")
  .user("What is the capital of France? Answer in one sentence.")
  .build()

// Stream the response
const program = Effect.gen(function* () {
  const responseStream = yield* model.stream(prompt, [], { maxTokens: 256 })

  yield* Stream.runForEach(responseStream, (event) =>
    Effect.sync(() => {
      switch (event._tag) {
        case "message_delta":
          process.stdout.write(event.text)
          break
        case "response_done":
          process.stdout.write(`\n\nDone: ${event.reason}\n`)
          break
      }
    }),
  )
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise)
