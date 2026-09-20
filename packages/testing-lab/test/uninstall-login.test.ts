import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { verifyRemovedLogin } from "../src/suites/uninstall-login"
import { sha256 } from "../src/snapshot"
import { ProcessExecutor, ProcessExecutorLive } from "../src/process"

for (const mode of ["absent", "dormant", "still-runnable", "changed-preference", "dangling-link"] as const) test(`uninstall login verification distinguishes ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-login-removal-" }).pipe(Effect.flatMap(fs.realPath))
  const path = join(root, "autostart.desktop"), executable = join(root, "app")
  const wire = "owned enabled login entry guarded by the captured executable"
  if (mode === "dangling-link") yield* fs.symlink(join(root, "missing"), path)
  else if (mode !== "absent") yield* fs.writeFileString(path, mode === "changed-preference" ? "changed" : wire)
  if (mode === "still-runnable") yield* fs.writeFileString(executable, "app")
  const result = yield* verifyRemovedLogin({ _tag: "Linux", path, executable, sha256: sha256(wire) }).pipe(Effect.either)
  expect(result._tag).toBe(mode === "absent" || mode === "dormant" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.state).toBe(mode === "absent" ? "Absent" : "Dormant")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))

for (const mode of ["absent", "dormant", "still-runnable", "changed-command", "unreadable"] as const) test(`Windows login removal distinguishes ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-windows-login-" })
  const executable = join(root, "Magnitude.exe")
  if (mode === "still-runnable") yield* fs.writeFileString(executable, "fixture")
  const value = `"${executable}" --background`
  const executor = Layer.succeed(ProcessExecutor, { run: spec => {
    expect(spec.executable).toBe("powershell.exe")
    expect(spec.env.LAB_LOGIN_KEY).toBe("HKCU:\\Software\\OwnedFixture")
    return Effect.succeed({ stdout: JSON.stringify({ command: mode === "absent" ? null : mode === "changed-command" ? "another.exe" : value }),
      stderr: "", exitCode: mode === "unreadable" ? 1 : 0 })
  } })
  const result = yield* verifyRemovedLogin({ _tag: "Windows", path: "HKCU:\\Software\\OwnedFixture", executable, sha256: sha256(value) }).pipe(Effect.either, Effect.provide(executor))
  expect(result._tag).toBe(mode === "absent" || mode === "dormant" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.state).toBe(mode === "absent" ? "Absent" : "Dormant")
})).pipe(Effect.provide(BunContext.layer))))

for (const mode of ["removed", "bundle-remains", "dangling-bundle"] as const) test(`macOS login removal distinguishes ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-mac-login-" })
  const path = join(root, "Magnitude.app"), executable = join(path, "Contents/MacOS/Magnitude")
  if (mode === "bundle-remains") yield* fs.makeDirectory(path)
  if (mode === "dangling-bundle") yield* fs.symlink(join(root, "missing"), path)
  const result = yield* verifyRemovedLogin({ _tag: "MacOS", path, registration: { executable, status: "enabled", packaged: true } }).pipe(Effect.either)
  expect(result._tag).toBe(mode === "removed" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right.state).toBe("Unlaunchable")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
