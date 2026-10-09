import { BunContext } from "@effect/platform-bun"
import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { macStartupUpdateOperation } from "./mac-startup-update"
const run = (receipt: boolean) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped()
  if (receipt) {
    yield* fs.makeDirectory(join(root, ".Magnitude.app.update"))
    yield* fs.writeFileString(join(root, ".Magnitude.app.update/transaction.json"), "receipt")
  }
  return yield* macStartupUpdateOperation(join(root, "Magnitude.app"))
})).pipe(Effect.provide(BunContext.layer)))
describe("macOS startup update selection", () => {
  it("never installs a prepared update at startup", async () => {
    expect(await run(false)).toEqual(Option.none())
  })
  it("recovers a published receipt", async () => {
    expect(await run(true)).toEqual(Option.some("Recover"))
  })
})
