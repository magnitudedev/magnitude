import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema } from "effect"
import { resolve } from "node:path"
import { Digest, InvalidInput } from "../src/domain"
import { LinuxInitialization, linuxInitialization } from "../src/providers/linux-initialization"
import { sha256 } from "../src/snapshot"

BunRuntime.runMain(Effect.gen(function* () {
  const [input, output] = process.argv.slice(2)
  if (!input || !output || process.argv.length !== 4) return yield* new InvalidInput({ message: "Expected initialization JSON and a new cloud-init output file" })
  const fs = yield* FileSystem.FileSystem
  const config = yield* fs.readFileString(input).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(LinuxInitialization))))
  const contents = yield* linuxInitialization(config)
  const file = resolve(output)
  yield* fs.writeFileString(file, contents, { mode: 0o600, flag: "wx" })
  // URLs may contain narrowly scoped download capabilities. Only the pin belongs in server config or logs.
  yield* Console.log(yield* Schema.encode(Schema.parseJson(Schema.Struct({ file: Schema.String, sha256: Digest })))({ file, sha256: sha256(contents) }))
}).pipe(Effect.provide(BunContext.layer)))
