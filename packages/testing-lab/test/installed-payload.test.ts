import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { installedPayload, verifyInstalledPayload } from "../src/suites/installed-payload"

for (const mutation of ["none", "bytes", "missing", "extra", "link", "version", "installer"] as const) {
  test(`updater payload compares against independent clean installation: ${mutation}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-payload-" })
    const file = join(root, "Magnitude.exe")
    yield* fs.writeFileString(file, "candidate executable")
    yield* fs.symlink("Magnitude.exe", join(root, "alias"))
    const candidate = yield* Schema.decodeUnknown(Candidate)({ target: targets.find(t => t.id === "windows-server-2025-x64-cpu-intel")!,
      version: "1.2.4", path: "/candidate.exe", artifact: { id: "desktop-windows-x64", kind: "desktop", host: "windows-x64-msvc",
        filename: "Magnitude.exe", bytes: 1, sha256: "a".repeat(64) } })
    const app = InstalledApplication.make({ candidate, root, executable: file, cli: file, packageVersion: "1.2.4" })
    const expected = yield* installedPayload(app)
    if (mutation === "bytes") yield* fs.writeFileString(file, "wrong executable")
    if (mutation === "missing") yield* fs.remove(file)
    if (mutation === "extra") yield* fs.writeFileString(join(root, "unexpected.dll"), "extra")
    if (mutation === "link") {
      yield* fs.remove(join(root, "alias"))
      yield* fs.symlink("elsewhere", join(root, "alias"))
    }
    const observed = mutation === "version" ? { ...app, candidate: { ...candidate, version: "1.2.3" } }
      : mutation === "installer" ? { ...app, candidate: { ...candidate, artifact: { ...candidate.artifact, sha256: "b".repeat(64) } } } : app
    const result = yield* verifyInstalledPayload(observed, expected).pipe(Effect.either)
    expect(result._tag).toBe(mutation === "none" ? "Right" : "Left")
    if (result._tag === "Right") expect(result.right.files).toHaveLength(1)
  })).pipe(Effect.provide(BunContext.layer))))
}
