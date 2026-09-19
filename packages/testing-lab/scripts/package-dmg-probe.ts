import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect } from "effect"
import { packageDesktopDmg } from "../../release/scripts/apple/desktop"

const run = Effect.gen(function* () {
  const app = yield* Config.string("LAB_PROBE_APP")
  const output = yield* Config.string("LAB_PROBE_DMG")
  yield* packageDesktopDmg(app, output)
  yield* Effect.logInfo(`Created installer ${output}`)
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
