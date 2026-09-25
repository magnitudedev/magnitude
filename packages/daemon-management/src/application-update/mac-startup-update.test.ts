import { BunContext } from "@effect/platform-bun"
import { FileSystem } from "@effect/platform"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { PreparedUpdateStore, type PreparedUpdate } from "../desktop-native/prepared-update"
import { macStartupUpdateOperation } from "./mac-startup-update"
const run = (state: "None" | "Unattempted" | "Attempted" | "Failed", receipt: boolean) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped()
  if (receipt) {
    yield* fs.makeDirectory(join(root, ".Magnitude.app.update"))
    yield* fs.writeFileString(join(root, ".Magnitude.app.update/transaction.json"), "receipt")
  }
  const pending = state === "None" ? Option.none() : Option.some({ release: { version: "0.1.6", bytes: 1, sha256: "0".repeat(64), signature: "fixture" },
    installation: state === "Failed" ? { _tag: state, reason: "Interrupted" } : { _tag: state } } as PreparedUpdate)
  return yield* macStartupUpdateOperation(join(root, "Magnitude.app"), "0.1.5").pipe(
    Effect.provideService(PreparedUpdateStore, { read: Effect.succeed(pending), removeAbandonedTransfers: Effect.void,
      discard: Effect.die("Unexpected discard"), prepare: () => Effect.die("Unexpected prepare"),
      verify: () => Effect.die("Unexpected verify"), recordAttempt: () => Effect.die("Unexpected attempt"),
      recordFailure: () => Effect.die("Unexpected failure") }))
})).pipe(Effect.provide(BunContext.layer)))
describe("macOS startup update selection", () => {
  it.each(["None", "Attempted", "Failed"] as const)("does not install %s preparation", async state => {
    expect(await run(state, false)).toEqual(Option.none())
  })
  it("installs unattempted preparation with the foreground invocation", async () => {
    expect(await run("Unattempted", false)).toEqual(Option.some("Install"))
  })
  it("recovers a published receipt even without prepared state", async () => {
    expect(await run("None", true)).toEqual(Option.some("Recover"))
  })
  it("recovers before considering another prepared attempt", async () => {
    expect(await run("Unattempted", true)).toEqual(Option.some("Recover"))
  })
})
