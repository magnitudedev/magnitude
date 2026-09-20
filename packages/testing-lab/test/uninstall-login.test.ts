import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { verifyRemovedLogin } from "../src/suites/uninstall-login"
import { sha256 } from "../src/snapshot"

for (const mode of ["absent", "dormant", "still-runnable", "changed-preference", "dangling-link"] as const) test(`uninstall login verification distinguishes ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-login-removal-" }).pipe(Effect.flatMap(fs.realPath))
  const path = join(root, "autostart.desktop"), executable = join(root, "app")
  const wire = "owned enabled login entry guarded by the captured executable"
  if (mode === "dangling-link") yield* fs.symlink(join(root, "missing"), path)
  else if (mode !== "absent") yield* fs.writeFileString(path, mode === "changed-preference" ? "changed" : wire)
  if (mode === "still-runnable") yield* fs.writeFileString(executable, "app")
  const result = yield* verifyRemovedLogin({ path, executable, sha256: sha256(wire) }).pipe(Effect.either)
  expect(result._tag).toBe(mode === "absent" || mode === "dormant" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.state).toBe(mode === "absent" ? "Absent" : "Dormant")
})).pipe(Effect.provide(BunContext.layer))))
