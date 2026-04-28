import { FetchHttpClient } from "@effect/platform"
import { Schema } from "effect"
import { Effect, Layer, Stream } from "effect"
import {
  AiTracer,
  bindModel,
  defineTool,
  execute,
  getProvider,
  NoopAiTracer,
  PromptBuilder,
  resolveEnvAuth,
} from "../src/index.js"

const weatherTool = defineTool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: Schema.Struct({
    city: Schema.String,
  }),
  outputSchema: Schema.String,
})

const provider = getProvider("openai")
if (!provider) {
  throw new Error('Provider "openai" not found')
}

const model =
  provider.models.find((candidate) => candidate.supportsToolCalls) ?? provider.models[0]
if (!model) {
  throw new Error('No models found for provider "openai"')
}

const auth = resolveEnvAuth(provider)
if (!auth) {
  throw new Error("Set OPENAI_API_KEY before running this example")
}

const prompt = PromptBuilder.empty()
  .system("You may call tools when they are useful.")
  .user("What's the weather in Tokyo right now?")
  .build()

const program = execute(bindModel(provider, model, auth), prompt, [weatherTool], {}).pipe(
  Stream.runForEach((event) =>
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
  ),
)

const ExampleLive = Layer.mergeAll(
  FetchHttpClient.layer,
  Layer.succeed(AiTracer, NoopAiTracer),
)

program.pipe(Effect.provide(ExampleLive), Effect.runPromise)
