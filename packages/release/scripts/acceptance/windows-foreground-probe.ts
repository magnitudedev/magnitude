import { BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Exit, Schema } from "effect"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"

/** Keep the installed compiled runtime and native addon mapped during replacement. */
const probe = Effect.gen(function* () {
  yield* Effect.sync(() => {
    const native = createRequire(import.meta.url)(join(dirname(process.execPath), "desktop-host.node")) as {
      readonly localAppDataDirectory: () => string
    }
    if (!native.localAppDataDirectory()) throw new Error("Native installation lookup failed")
  })
  yield* Console.log("ready")
  if (process.argv.includes("--launcher-probe")) {
    yield* Console.log(yield* Schema.encode(Schema.parseJson(Schema.Struct({
      args: Schema.Array(Schema.String), cwd: Schema.String,
    })))({ args: process.argv.slice(2), cwd: process.cwd() }))
  }
  const code = yield* Effect.async<number>(resume => {
    const end = () => resume(Effect.succeed(0))
    const data = () => resume(Effect.succeed(75))
    if (process.argv.includes("--launcher-probe")) process.stdin.once("data", data)
    process.stdin.once("end", end)
    process.stdin.resume()
    return Effect.sync(() => { process.stdin.removeListener("end", end); process.stdin.removeListener("data", data); process.stdin.pause() })
  })
  return code
})
BunRuntime.runMain(probe, {
  teardown: (exit, onExit) => onExit(Exit.isSuccess(exit) && typeof exit.value === "number" ? exit.value : 1),
})
