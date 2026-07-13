import { Effect, Schema, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  Auth,
  defineTool,
  NativeChatCompletions,
  PromptBuilder,
} from "@magnitudedev/ai"
import {
  defineHarnessTool,
  defineToolkit,
  createHarness,
} from "../src/index.js"

// ── Define a tool ────────────────────────────────────────────────────

const weatherDef = defineTool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: Schema.Struct({ city: Schema.String }),
  outputSchema: Schema.String,
})

const weatherTool = defineHarnessTool({
  definition: weatherDef,
  execute: (input) =>
    Effect.succeed(`It's 72°F and sunny in ${input.city}.`),
})

// ── Build toolkit ────────────────────────────────────────────────────

const toolkit = defineToolkit({
  weather: { tool: weatherTool },
})

// ── Bind model ───────────────────────────────────────────────────────

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) throw new Error("Set OPENAI_API_KEY before running this example")

const openaiGpt4o = NativeChatCompletions.model({
  id: "openai/gpt-4o",
  modelId: "gpt-4o",
  endpoint: "https://api.openai.com/v1",
  contextWindow: 128_000,
  maxOutputTokens: 16_384,
  options: NativeChatCompletions.options,
})

const model = openaiGpt4o.bind({ auth: Auth.bearer(apiKey) })

// ── Create harness and run ───────────────────────────────────────────

const harness = createHarness({ model, toolkit })

const prompt = PromptBuilder.empty()
  .system("You have access to a weather tool. Use it when asked about weather.")
  .user("What's the weather in Tokyo?")
  .build()

const program = Effect.gen(function* () {
  const turn = yield* harness.runTurn(prompt)

  yield* Stream.runForEach(turn.events, (event) =>
    Effect.sync(() => {
      switch (event._tag) {
        case "MessageDelta":
          process.stdout.write(event.text)
          break
        case "ToolInputStarted":
          console.log(`\n[tool] ${event.toolName} started`)
          break
        case "ToolExecutionEnded":
          console.log(`[tool] ${event.toolName} → ${event.result._tag}`)
          break
        case "TurnEnd":
          console.log(`\n[done] ${event.outcome._tag}`)
          break
      }
    }),
  )
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise)
