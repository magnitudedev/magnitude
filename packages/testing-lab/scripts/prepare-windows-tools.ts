import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema } from "effect"
import { resolve } from "node:path"
import { Digest, InvalidInput } from "../src/domain"
import { windowsToolPreparation } from "../src/providers/windows-tools"
import { sha256 } from "../src/snapshot"

BunRuntime.runMain(Effect.gen(function* () {
  const [output] = process.argv.slice(2)
  if (!output || process.argv.length !== 3) return yield* new InvalidInput({ message: "Expected a new Windows tool preparation script path" })
  const fs = yield* FileSystem.FileSystem
  const contents = yield* windowsToolPreparation
  const file = resolve(output)
  yield* fs.writeFileString(file, contents, { mode: 0o600, flag: "wx" })
  yield* Console.log(yield* Schema.encode(Schema.parseJson(Schema.Struct({ file: Schema.String, sha256: Digest })))({ file, sha256: sha256(contents) }))
}).pipe(Effect.provide(BunContext.layer)))
