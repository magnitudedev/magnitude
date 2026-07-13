import { Effect, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  Auth,
  Model,
  NativeChatCompletions,
  Option,
  PromptBuilder,
} from "../src/index.js"

// ---------------------------------------------------------------------------
// 1. Custom option: reasoning effort
// ---------------------------------------------------------------------------
const fireworksMinimax = NativeChatCompletions.model({
  id: "fireworks/minimax-m2.7",
  modelId: "accounts/fireworks/models/minimax-m2.7",
  endpoint: "https://api.fireworks.ai/inference/v1",
  contextWindow: 196_000,
  maxOutputTokens: 196_000,
  options: {
    ...NativeChatCompletions.options,
    reasoningEffort: Option.define(
      (val: "none" | "low" | "medium") => ({ reasoning_effort: val }),
    ),
  },

})

// ---------------------------------------------------------------------------
// 2. Cross-cutting concern with compose
// ---------------------------------------------------------------------------
type DeepSeekThinking = { type: "enabled" } | { type: "disabled" }

const deepseekV4 = NativeChatCompletions.model({
  id: "deepseek/deepseek-v4-pro",
  modelId: "deepseek-v4-pro",
  endpoint: "https://api.deepseek.com/v1",
  contextWindow: 262_144,
  maxOutputTokens: 131_072,
  options: {
    ...NativeChatCompletions.options,
    thinking: Option.define(
      (val: DeepSeekThinking) => ({ thinking: val }),
    ),
  },
  compose: (wire, callOpts) => {
    if (callOpts.thinking?.type === "enabled") {
      return { ...wire, temperature: undefined }
    }
    return wire
  },

})

// ---------------------------------------------------------------------------
// Demo: use the Fireworks model with reasoning effort
// ---------------------------------------------------------------------------
const apiKey = process.env.FIREWORKS_API_KEY
if (!apiKey) throw new Error("Set FIREWORKS_API_KEY before running this example")

const model = fireworksMinimax.bind({ auth: Auth.bearer(apiKey) })

const prompt = PromptBuilder.empty()
  .system("You are a helpful assistant.")
  .user("Explain quantum entanglement simply.")
  .build()

const program = Effect.gen(function* () {
  const responseStream = yield* model.stream(prompt, [], {
    maxTokens: 1000,
    reasoningEffort: "low",
  })

  yield* Stream.runForEach(responseStream, (event) =>
    Effect.sync(() => {
      if (event._tag === "message_delta") process.stdout.write(event.text)
      if (event._tag === "response_done") console.log(`\n\nDone: ${event.reason}`)
    }),
  )
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise)
