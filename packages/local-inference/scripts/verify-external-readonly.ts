import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { Console, Data, Effect, Layer, Option, Redacted, Schema } from "effect"
import { LlamaInstanceId, makeLlamaServerClient } from "../src/llamacpp"

class MissingOrigin extends Data.TaggedError("MissingOrigin")<{}> {}
class InvalidOrigin extends Data.TaggedError("InvalidOrigin")<{ readonly value: string }> {}
const HttpOrigin = Schema.URL.pipe(Schema.filter((url) => url.protocol === "http:" || url.protocol === "https:", { message: () => "Expected an HTTP(S) URL" }))
const JsonOutput = Schema.parseJson(Schema.Unknown, { space: 2 })

const program = Effect.gen(function* () {
  const rawOrigin = process.argv[2]
  if (rawOrigin === undefined) return yield* new MissingOrigin()
  const origin = yield* Schema.decodeUnknown(HttpOrigin)(rawOrigin).pipe(Effect.mapError(() => new InvalidOrigin({ value: rawOrigin })))
  const authorization = Option.map(Option.fromNullable(process.env.LLAMA_API_KEY), Redacted.make)
  const client = yield* makeLlamaServerClient({ origin, authorization, timeout: Option.none() })
  const observation = yield* client.observer.observe(LlamaInstanceId.make(`manual_${createHash("sha256").update(origin.origin).digest("hex")}`), "external")
  yield* Schema.encode(JsonOutput)({ verifiedAt: new Date().toISOString(), observation, mutationAttempted: false }).pipe(Effect.flatMap(Console.log))
})

BunRuntime.runMain(program.pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer))))
import { createHash } from "node:crypto"
