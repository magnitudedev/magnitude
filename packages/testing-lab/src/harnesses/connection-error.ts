import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { basename, join } from "node:path"
import { AssertionFailure, Harness } from "../domain"
import { DesktopDriver } from "../desktop-driver"

/** Fault only an existing file in the isolated fixture, and restore its bytes even on failure. */
export const exerciseConnectionError = (isolatedHome: string, harness: typeof Harness.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const desktop = yield* DesktopDriver
  const file = join(isolatedHome, harness === "pi" ? ".pi/agent/models.json" : harness === "opencode" ? ".config/opencode/opencode.json" : ".hermes/config.yaml")
  const malformed = '{"providers": [ '
  const diagnostic = yield* Effect.acquireUseRelease(fs.readFile(file), () => Effect.gen(function* () {
    yield* fs.writeFileString(file, malformed)
    const message = yield* desktop.connectionFailure(harness, basename(file))
    if ((yield* fs.readFileString(file)) !== malformed) return yield* new AssertionFailure({ message: `${harness} error handling overwrote the malformed configuration` })
    return message
  }), original => fs.writeFile(file, original).pipe(Effect.orDie))
  yield* desktop.connect(harness)
  return diagnostic
})
