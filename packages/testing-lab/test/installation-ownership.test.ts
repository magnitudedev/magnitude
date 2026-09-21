import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { ApplicationIdentity } from "../src/application-identity"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { CliTests } from "../src/suites/cli"
import { verifyInstallationOwnership } from "../src/suites/installation-ownership"

for (const targetId of ["ubuntu-24.04-x64-cpu-intel", "windows-server-2022-x64-cpu-intel", "windows-server-2025-x64-cpu-intel"]) {
for (const mode of ["owned", "symlink", "foreign-entrypoint", "escaped-bundle", "changed-owner"] as const) test(`installation ownership rejects substitution: ${targetId}/${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-install-ownership-" })
  const target = targets.find(target => target.id === targetId)!
  const windows = target.os === "windows-server", filename = windows ? "app.exe" : "app.deb"
  const root = join(temporary, "app"), bundled = join(root, "resources", windows ? "magnitude.exe" : "magnitude"), outside = join(temporary, "other-cli")
  yield* fs.makeDirectory(join(root, "resources"), { recursive: true })
  yield* fs.writeFileString(outside, "another CLI")
  if (mode === "escaped-bundle") yield* fs.symlink(outside, bundled)
  else yield* fs.writeFileString(bundled, "owned CLI")
  const cli = mode === "symlink" ? join(temporary, "launcher") : mode === "foreign-entrypoint" ? outside : bundled
  if (mode === "symlink") yield* fs.symlink(bundled, cli)
  const app = yield* Schema.decodeUnknown(InstalledApplication)({ root, executable: join(root, "magnitude"), cli, packageVersion: "0.1.3",
    candidate: { version: "0.1.3", target, path: join(temporary, filename),
      artifact: { id: "desktop", kind: "desktop", host: target.artifactHost, filename, bytes: 1, sha256: "a".repeat(64) } } })
  const owner = yield* Schema.decodeUnknown(ApplicationIdentity)({ applicationPid: 101, servicePid: 102, serviceInstance: "first" })
  let observed = 0, invoked = 0
  const result = yield* verifyInstallationOwnership(app, { ready: () => Effect.void, identity: () => Effect.sync(() => {
    observed++
    return mode === "changed-owner" && observed > 1 ? { ...owner, serviceInstance: Schema.decodeUnknownSync(ApplicationIdentity.fields.serviceInstance)("replacement") } : owner
  }) }).pipe(Effect.provide(Layer.succeed(CliTests, {
    version: Effect.sync(() => { invoked++ }), ensureService: Effect.sync(() => { invoked++ }),
    inspect: Effect.void, reloadModel: Effect.void, loadModel: Effect.void, removeModel: Effect.void, failedModel: Effect.void, modelLifecycle: Effect.void, connections: () => Effect.void, invalid: Effect.void, nativeRuntime: Effect.void,
  })), Effect.either)
  expect(result._tag).toBe(mode === "owned" || mode === "symlink" ? "Right" : "Left")
  expect(invoked).toBe(mode === "foreign-entrypoint" || mode === "escaped-bundle" ? 0 : 2)
  if (result._tag === "Right") expect(result.right.before).toEqual(result.right.after)
})).pipe(Effect.provide(BunContext.layer))))
}
