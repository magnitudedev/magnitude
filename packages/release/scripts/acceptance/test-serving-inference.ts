import { FetchHttpClient, FileSystem, HttpClient, HttpClientRequest, HttpClientResponse } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Chunk, Config, Effect, Schema, Stream } from "effect"

// Run against an isolated, already serving application with a catalog model installed.
class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Completion = Schema.Struct({
  choices: Schema.NonEmptyArray(Schema.Struct({ message: Schema.Struct({ content: Schema.NullOr(Schema.String) }) })),
  usage: Schema.Struct({ completion_tokens: Schema.Number }),
})
const StreamChunk = Schema.Struct({ choices: Schema.Array(Schema.Struct({
  delta: Schema.Struct({ content: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }) }),
})) })
const Evidence = Schema.Struct({ model: Schema.String, baseline: Completion, cancelledStreamChunks: Schema.Number, afterCancellation: Completion })
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const model = yield* Config.string("MAGNITUDE_ACCEPTANCE_MODEL")
  const origin = yield* Config.string("MAGNITUDE_ACCEPTANCE_ORIGIN")
  const output = yield* Config.string("MAGNITUDE_ACCEPTANCE_RESULT")
  const client = (yield* HttpClient.HttpClient).pipe(HttpClient.filterStatusOk)
  const request = (prompt: string, stream = false) => HttpClientRequest.post(`${origin}/inference/v1/chat/completions`).pipe(
    HttpClientRequest.bodyJson({ model, messages: [{ role: "user", content: prompt }],
      max_tokens: 160, reasoning_effort: "none", stream }),
    Effect.flatMap(client.execute),
  )
  const generate = Effect.scoped(Effect.gen(function* () {
    const response = yield* request("Reply with the single word ready.")
    const result = yield* HttpClientResponse.schemaBodyJson(Completion)(response)
    if (result.usage.completion_tokens <= 0 || !result.choices[0].message.content?.trim()) {
      return yield* new AcceptanceFailed({ message: "Generation returned no text or completion tokens" })
    }
    return result
  })).pipe(Effect.timeout("5 minutes"))
  const baseline = yield* generate
  const chunks = yield* Effect.scoped(Effect.gen(function* () {
    const response = yield* request("Count from one to one hundred, spelling each number.", true)
    return yield* response.stream.pipe(
      Stream.decodeText(), Stream.splitLines,
      Stream.filter(line => line.startsWith("data: {") ),
      Stream.mapEffect(line => Schema.decode(Schema.parseJson(StreamChunk))(line.slice(6))),
      Stream.filter(chunk => chunk.choices.length > 0), Stream.take(8), Stream.runCollect,
    )
  })).pipe(Effect.timeout("5 minutes"))
  if (Chunk.size(chunks) !== 8) return yield* new AcceptanceFailed({ message: "Stream ended before cancellation boundary" })
  const afterCancellation = yield* generate
  yield* fs.writeFileString(output, yield* Schema.encode(Schema.parseJson(Evidence))({
    model, baseline, cancelledStreamChunks: Chunk.size(chunks), afterCancellation,
  }))
  yield* Effect.logInfo("Generation, stream cancellation and subsequent generation passed")
})
BunRuntime.runMain(run.pipe(Effect.provide([FetchHttpClient.layer, BunContext.layer])))
