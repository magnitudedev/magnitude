import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema } from "effect"
import { resolve } from "node:path"
import { InvalidInput } from "../src/domain"
import { WindowsRuntimePreparation, windowsRuntimePreparation } from "../src/providers/windows-initialization"

// Writes a private native-command request, not a script containing the download capability.
BunRuntime.runMain(Effect.gen(function* () {
  const [input, output, location] = process.argv.slice(2)
  if (!input || !output || !location || process.argv.length !== 5 || !/^[a-z0-9]+$/.test(location)) return yield* new InvalidInput({ message: "Expected runtime configuration, a new request path and Azure location" })
  const fs = yield* FileSystem.FileSystem
  const configuration = yield* fs.readFileString(input).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WindowsRuntimePreparation))),
    Effect.mapError(() => new InvalidInput({ message: "Windows runtime configuration is invalid or unreadable" })))
  const encoded = yield* Schema.encode(Schema.parseJson(WindowsRuntimePreparation))(configuration)
  const Request = Schema.Struct({ location: Schema.String, properties: Schema.Struct({ source: Schema.Struct({ script: Schema.String }),
    asyncExecution: Schema.Literal(true), timeoutInSeconds: Schema.Literal(2400), protectedParameters: Schema.Array(Schema.Struct({ name: Schema.Literal("LAB_INITIALIZATION"), value: Schema.String })) }) })
  yield* fs.writeFileString(resolve(output), yield* Schema.encode(Schema.parseJson(Request))({ location, properties: {
    source: { script: yield* windowsRuntimePreparation }, asyncExecution: true, timeoutInSeconds: 2400,
    protectedParameters: [{ name: "LAB_INITIALIZATION", value: Buffer.from(encoded).toString("base64") }],
  } }), { mode: 0o600, flag: "wx" })
  yield* Console.log("Prepared private Windows runtime request; native readiness remains unverified")
}).pipe(Effect.provide(BunContext.layer)))
