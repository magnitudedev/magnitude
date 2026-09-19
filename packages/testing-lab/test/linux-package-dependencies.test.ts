import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { ArchiveExtractor } from "../../release/src/archive"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"
import { inspectLinuxPackageDependencies } from "../src/suites/linux-package-dependencies"

test("Linux package inspection covers installed and admitted runtime bytes and rejects a missing dynamic module dependency", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-linux-package-" })
  const application = join(root, "app")
  yield* fs.makeDirectory(application)
  const native = new Uint8Array(64), view = new DataView(native.buffer)
  native.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]); view.setUint16(18, 62, true)
  for (const name of ["magnitude", "bridge.node"]) yield* fs.writeFile(join(application, name), native)
  const launcher = join(application, "launch")
  yield* fs.writeFileString(launcher, '#!/bin/sh\nexec "$(dirname "$0")/magnitude" "$@"\n')
  const payload = new TextEncoder().encode("verified fixture archive")
  const release = yield* Schema.decodeUnknown(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
    rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [
      { id: "desktop", kind: "desktop", host: "linux-x64-gnu", filename: "app.deb", bytes: payload.length, sha256: sha256(payload) },
      { id: "base", kind: "icn-base", host: "linux-x64-gnu", backend: "cpu", filename: "base.tar.gz", bytes: payload.length,
        sha256: sha256(payload), nativeBuild: "fixture", backendModuleAbi: "fixture" },
    ] })
  const target = targets.find(target => target.id === "ubuntu-24.04-x64-cpu-intel")!
  const app = InstalledApplication.make({ root: application, executable: launcher, cli: join(application, "magnitude"),
    packageVersion: "0.1.3", candidate: { version: "0.1.3", target, artifact: release.artifacts[0]!, path: join(root, "app.deb") } })
  let missing = false
  const inspected: string[] = []
  yield* Effect.gen(function* () {
    yield* (yield* ArtifactStore).put(sha256(payload), Stream.make(payload))
    const run = inspectLinuxPackageDependencies(app, release).pipe(Effect.provide([
      Layer.succeed(ArchiveExtractor, { extract: (_archive, destination) => Effect.gen(function* () {
        for (const directory of ["bin", "backends", "catalog"]) yield* fs.makeDirectory(join(destination, directory), { recursive: true })
        yield* fs.writeFile(join(destination, "bin", "magnitude-inference"), native)
        yield* fs.writeFile(join(destination, "backends", "cpu.so"), native)
        yield* fs.writeFileString(join(destination, "catalog", "model-planner-inputs.bundle"), "fixture")
      }).pipe(Effect.orDie) }),
      Layer.succeed(ProcessExecutor, { run: spec => {
        const file = spec.args.at(-1)!
        if (spec.args.includes("--program-headers")) {
          inspected.push(file)
          return Effect.succeed({ exitCode: 0, stderr: "", stdout: "Elf file type is DYN (Shared object file)\nProgram Headers:\n LOAD 0x0000\n" })
        }
        if (spec.args.includes("--version-info")) return Effect.succeed({ exitCode: 0, stderr: "", stdout: "No version information found in this file.\n" })
        return Effect.succeed({ exitCode: 0, stderr: "", stdout: `Dynamic section at offset 0x100 contains 1 entries:\n${missing && file.endsWith("cpu.so") ? " 0x1 (NEEDED) Shared library: [libmissing.so]\n" : ""} 0x0 (NULL) 0x0\n` })
      } }),
    ]))
    const report = yield* run
    expect(report.application.files).toHaveLength(2)
    expect(report.runtime.files).toHaveLength(2)
    expect(report.interpreters).toHaveLength(4)
    expect(inspected).toHaveLength(4)
    expect(yield* fs.exists(report.runtime.root)).toBe(false)
    missing = true
    const rejected = yield* run.pipe(Effect.either)
    expect(rejected._tag).toBe("Left")
    if (rejected._tag === "Left") expect(rejected.left.message).toContain("not an admitted OS dependency: libmissing.so")
    missing = false
    yield* fs.writeFileString(join(application, "magnitude"), "not a native executable")
    const invalidEntrypoint = yield* run.pipe(Effect.either)
    expect(invalidEntrypoint._tag).toBe("Left")
    if (invalidEntrypoint._tag === "Left") expect(invalidEntrypoint.left.message).toContain("Declared executable is absent from native inventory")
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))
