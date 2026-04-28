import { Effect, Stream } from "effect"
import {
  bindModel,
  execute,
  getProvider,
  PromptBuilder,
  resolveEnvAuth,
} from "../src/index.js"

const provider = getProvider("fireworks-ai")
if (!provider) {
  throw new Error('Provider "fireworks-ai" not found')
}

const model = provider.models[0]
if (!model) {
  throw new Error('No models found for provider "fireworks-ai"')
}

const auth = resolveEnvAuth(provider)
if (!auth) {
  throw new Error("Set FIREWORKS_API_KEY before running this example")
}

const prompt = PromptBuilder.empty()
  .system("You are a concise, helpful assistant.")
  .user("What is the capital of France? Answer in one sentence.")
  .build()

const program = execute(bindModel(provider, model, auth), prompt, [], {}).pipe(
  Stream.runForEach((event) =>
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
  ),
)

Effect.runPromise(program)
