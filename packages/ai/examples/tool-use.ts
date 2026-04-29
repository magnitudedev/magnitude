import { Effect, Schema, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  Auth,
  defineTool,
  Model,
  NativeChatCompletions,
  PromptBuilder,
} from "../src/index.js"

const weatherTool = defineTool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: Schema.Struct({ city: Schema.String }),
  outputSchema: Schema.String,
})

const openaiGpt4o = NativeChatCompletions.model({
  id: "openai/gpt-4o",
  modelId: "gpt-4o",
  endpoint: "https://api.openai.com/v1",
  contextWindow: 128_000,
  maxOutputTokens: 16_384,
  options: NativeChatCompletions.options,
})

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) throw new Error("Set OPENAI_API_KEY before running this example")

const model = openaiGpt4o.bind({ auth: Auth.bearer(apiKey) })

const prompt = PromptBuilder.empty()
  .system("You may call tools when they are useful.")
  .user("What's the weather in Tokyo right now?")
  .build()

const program = Effect.gen(function* () {
  const responseStream = yield* model.stream(prompt, [weatherTool], {
    maxTokens: 1000,
  })

  yield* Stream.runForEach(responseStream, (event) =>
    Effect.sync(() => {
      switch (event._tag) {
        case "message_delta":
          process.stdout.write(event.text)
          break
        case "tool_call_start":
          console.log(`\n[tool call start] ${event.toolName} (${event.toolCallId})`)
          break
        case "tool_call_field_delta":
          console.log(
            `[tool input delta] ${event.path.join(".") || "<root>"} += ${JSON.stringify(event.delta)}`,
          )
          break
        case "tool_call_field_end":
          console.log(
            `[tool input complete] ${event.path.join(".") || "<root>"} = ${JSON.stringify(event.value)}`,
          )
          break
        case "tool_call_end":
          console.log(`[tool call end] ${event.toolCallId}`)
          break
        case "response_done":
          process.stdout.write(`\n\nDone: ${event.reason}\n`)
          break
      }
    }),
  )
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise)
