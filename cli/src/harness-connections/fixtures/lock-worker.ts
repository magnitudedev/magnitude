import { BunContext, BunRuntime } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import { Console, Effect } from "effect"
import { withConnectionLock } from "../lock"

const [file, mode] = process.argv.slice(2)
const program = Effect.gen(function* () {
  if (!file) return yield* Effect.die("Missing lock target")
  const fs = yield* FileSystem.FileSystem
  if (mode === "hold") return yield* withConnectionLock(file, Console.log("acquired").pipe(Effect.zipRight(Effect.never)))
  for (let i = 0; i < 10; i++) yield* withConnectionLock(file, Effect.gen(function* () {
    const current = Number(yield* fs.readFileString(file))
    yield* Effect.yieldNow()
    yield* fs.writeFileString(file, String(current + 1))
  }))
})
BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
