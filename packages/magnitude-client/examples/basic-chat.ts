import { Effect, Stream } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { createMagnitudeClient } from "../src/index.js"
import { PromptBuilder } from "@magnitudedev/ai"

const client = createMagnitudeClient()

const prompt = PromptBuilder.empty()
  .user("What is 2 + 2?")
  .build()

const program = Effect.gen(function* () {
  const stream = yield* client.role("leader").stream(prompt, [])

  yield* Stream.runForEach(stream, (event) =>
    Effect.sync(() => {
      if (event._tag === "message_delta") {
        process.stdout.write(event.text)
      }
    }),
  )

  console.log()
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise).catch(console.error)
