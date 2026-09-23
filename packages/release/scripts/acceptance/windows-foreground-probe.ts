import { BunRuntime } from "@effect/platform-bun"
import { Console, Effect } from "effect"
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
  yield* Effect.async<void>(resume => {
    const end = () => resume(Effect.void)
    process.stdin.once("end", end)
    process.stdin.resume()
    return Effect.sync(() => { process.stdin.removeListener("end", end); process.stdin.pause() })
  })
})
BunRuntime.runMain(probe)
