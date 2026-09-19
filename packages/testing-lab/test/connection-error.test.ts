import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { DesktopDriver } from "../src/desktop-driver"
import { AssertionFailure } from "../src/domain"
import { exerciseConnectionError } from "../src/harnesses/connection-error"

for (const behavior of ["preserve", "overwrite", "missing-alert"] as const) test(`restores the original fixture after ${behavior}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-connection-error-" })
  const directory = join(root, ".pi", "agent")
  yield* fs.makeDirectory(directory, { recursive: true })
  const file = join(directory, "models.json")
  const original = '{"providers":{"unrelated":{"value":1}}}\n'
  yield* fs.writeFileString(file, original)
  let recovered = false
  const driver: DesktopDriver = {
    updates: { action: () => Effect.die("Unused"), automatic: () => Effect.die("Unused"), wait: () => Effect.die("Unused") },
    identity: () => Effect.die("Not used by this fixture"), host: () => Effect.succeed("unused"), navigate: () => Effect.void, ready: () => Effect.void, serviceFailure: () => Effect.succeed("unused"),
    search: () => Effect.void, details: () => Effect.void, download: () => Effect.void,
    load: () => Effect.void, disconnect: () => Effect.void, theme: () => Effect.void,
    verifyTheme: () => Effect.void, screenshot: () => Effect.succeed("unused"), text: () => Effect.succeed("unused"),
    quit: () => Effect.void, chrome: () => Effect.void,
    connectionFailure: (_harness, name) => {
      expect(name).toBe("models.json")
      if (behavior === "missing-alert") return Effect.fail(new AssertionFailure({ message: "No alert" }))
      return (behavior === "overwrite" ? fs.writeFileString(file, "{}").pipe(Effect.orDie) : Effect.void).pipe(Effect.as("Cannot parse models.json"))
    },
    connect: () => Effect.sync(() => { recovered = true }),
  }
  const result = yield* exerciseConnectionError(root, "pi").pipe(Effect.provideService(DesktopDriver, driver), Effect.either)
  expect(result._tag).toBe(behavior === "preserve" ? "Right" : "Left")
  expect(yield* fs.readFileString(file)).toBe(original)
  expect(recovered).toBe(behavior === "preserve")
})).pipe(Effect.provide(BunContext.layer))))
