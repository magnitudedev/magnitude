import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { verifyRemovedPayload } from "../src/suites/uninstall"

test("detects leftover payloads and dangling CLI launchers", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-uninstall-" })
  const app = { root: join(root, "application"), executable: join(root, "application", "Magnitude"), cli: join(root, "magnitude") }
  yield* verifyRemovedPayload(app)
  yield* fs.makeDirectory(app.root)
  expect((yield* verifyRemovedPayload(app).pipe(Effect.either))._tag).toBe("Left")
  yield* fs.remove(app.root, { recursive: true })
  yield* fs.symlink(join(root, "removed-cli"), app.cli)
  expect(yield* fs.exists(app.cli)).toBe(false)
  expect((yield* verifyRemovedPayload(app).pipe(Effect.either))._tag).toBe("Left")
  yield* fs.remove(app.cli)
  yield* verifyRemovedPayload(app)
})).pipe(Effect.provide(BunContext.layer))))
