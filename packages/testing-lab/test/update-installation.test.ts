import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"
import { observeUpdatedInstallation } from "../src/suites/update-installation"
import { verifyDebPayload } from "../src/suites/package-payload"

const target = targets.find(value => value.id === "ubuntu-24.04-x64-cpu-intel")!
for (const mode of ["exact", "changed-file", "changed-link", "file-became-link", "directory-became-link", "changed-package"] as const) test(`updated payload comparison rejects divergence: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-payload-" }).pipe(Effect.flatMap(fs.realPath))
  const directory = join(root, "installed")
  yield* fs.makeDirectory(directory)
  yield* fs.writeFileString(join(directory, "file"), mode === "changed-file" ? "wrong" : "installed bytes")
  yield* fs.symlink(mode === "changed-link" ? "wrong" : "file", join(directory, "link"))
  if (mode === "file-became-link") {
    yield* fs.rename(join(directory, "file"), join(root, "outside-file"))
    yield* fs.symlink(join(root, "outside-file"), join(directory, "file"))
  }
  if (mode === "directory-became-link") {
    yield* fs.rename(directory, join(root, "outside-directory"))
    yield* fs.symlink(join(root, "outside-directory"), directory)
  }
  const path = join(root, "candidate.deb"), archive = "admitted package"
  yield* fs.writeFileString(path, mode === "changed-package" ? "mutated archive" : archive)
  const candidate = yield* Schema.decodeUnknown(Candidate)({ target, version: "1.2.4", path,
    artifact: { id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.deb", bytes: archive.length, sha256: sha256(archive) } })
  const app = InstalledApplication.make({ candidate, root: directory, executable: join(directory, "file"), cli: join(directory, "file"), packageVersion: "1.2.4-44" })
  let extracted = false
  const result = yield* verifyDebPayload(app).pipe(Effect.provideService(ProcessExecutor, {
    // Only the extraction boundary is synthetic; comparisons use actual files and links.
    run: spec => Effect.gen(function* () {
      expect(spec.executable).toBe("dpkg-deb")
      expect(spec.args.slice(0, 2)).toEqual(["-x", path])
      extracted = true
      const destination = join(spec.args[2]!, directory.slice(1))
      yield* fs.makeDirectory(destination, { recursive: true })
      yield* fs.writeFileString(join(destination, "file"), "installed bytes")
      yield* fs.symlink("file", join(destination, "link"))
      return { exitCode: 0, stdout: "", stderr: "" }
    }).pipe(Effect.orDie),
  }), Effect.either)
  expect(result._tag).toBe(mode === "exact" ? "Right" : "Left")
  expect(extracted).toBe(mode !== "changed-package")
  if (result._tag === "Right") expect(result.right.files).toEqual([{ path: join(directory, "file"), sha256: sha256("installed bytes"), bytes: 15 }])
})).pipe(Effect.provide(BunContext.layer))))

for (const mode of ["updated", "old-package", "old-cli"] as const) test(`native update observation never installs a package: ${mode}`, () => Effect.runPromise(Effect.gen(function* () {
  const candidate = yield* Schema.decodeUnknown(Candidate)({ target, version: "1.2.4", path: "/candidate.deb", artifact: {
    id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.deb", bytes: 1, sha256: "a".repeat(64) } })
  const previous = InstalledApplication.make({ candidate: { ...candidate, version: "1.2.3" }, root: "/app", executable: "/app/exe", cli: "/app/cli", packageVersion: "1.2.3-44" })
  const commands: string[] = []
  const result = yield* observeUpdatedInstallation(previous, candidate, {}).pipe(Effect.provideService(ProcessExecutor, {
    run: spec => Effect.sync(() => {
      commands.push(spec.executable)
      expect(["dpkg-deb", "dpkg-query", previous.cli]).toContain(spec.executable)
      return { exitCode: 0, stderr: "", stdout: spec.executable === previous.cli ? mode === "old-cli" ? "1.2.3" : "1.2.4"
        : spec.executable === "dpkg-query" && mode === "old-package" ? "1.2.3-44" : "1.2.4-44" }
    }),
  }), Effect.either)
  expect(result._tag).toBe(mode === "updated" ? "Right" : "Left")
  expect(commands).toEqual(mode === "old-package" ? ["dpkg-deb", "dpkg-query"] : ["dpkg-deb", "dpkg-query", previous.cli])
  if (result._tag === "Right") expect(result.right.packageVersion).toBe("1.2.4-44")
})))
