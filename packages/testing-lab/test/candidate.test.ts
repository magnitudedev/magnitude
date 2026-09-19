import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema, Stream } from "effect"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { join } from "node:path"
import releasePlan from "../../release/release-plan.json"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { Candidate, prepareCandidate, selectInstaller } from "../src/candidate"
import { Installer, nativeInstaller } from "../src/installer"
import { targets } from "../src/catalog"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"
import { rejectCorruptInstaller } from "../src/suites/install"
import { AssertionFailure } from "../src/domain"

const target = targets.find(t => t.id === "macos-15-arm64-metal-apple-silicon")!
const payload = new TextEncoder().encode("fixture installer bytes")
const manifest = Schema.decodeUnknownSync(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
  rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [{
    id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: payload.length, sha256: sha256(payload),
  }] })

test("candidate download binds exact format, host, digest and length", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-candidate-" })
  yield* Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(payload), Stream.make(payload))
    const candidate = yield* prepareCandidate(manifest, target, join(root, "download"))
    expect(candidate.artifact.sha256).toBe(sha256(payload))
    expect(yield* fs.readFileString(candidate.path)).toBe("fixture installer bytes")
    expect((yield* selectInstaller(manifest, { ...target, packageFormat: "rpm" }).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* selectInstaller({ ...manifest, artifacts: [manifest.artifacts[0], { ...manifest.artifacts[0], id: "duplicate", filename: "second.dmg" }] }, target).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* prepareCandidate({ ...manifest, artifacts: [{ ...manifest.artifacts[0], bytes: 1 }] }, target, join(root, "wrong-length")).pipe(Effect.either))._tag).toBe("Left")
    yield* fs.writeFileString(join(root, "objects", candidate.artifact.sha256), "corrupt object")
    expect((yield* prepareCandidate(manifest, target, join(root, "corrupt")).pipe(Effect.either))._tag).toBe("Left")
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))

test("native installer rejects altered bytes before executing any package command", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-install-rejection-" })
  const path = join(root, "Magnitude.dmg")
  yield* fs.writeFileString(path, "changed after download")
  let calls = 0
  const native = { ...target, os: process.platform === "darwin" ? "macos" as const : process.platform === "win32" ? "windows" as const : "ubuntu" as const,
    arch: process.arch === "arm64" ? "arm64" as const : "x64" as const }
  const candidate = Candidate.make({ artifact: manifest.artifacts[0], version: manifest.version, target: native, path })
  const result = yield* Effect.flatMap(Installer, installer => installer.install(candidate)).pipe(Effect.either,
    Effect.provide(nativeInstaller({ disposable: true, root: join(root, "app"), environment: {} }).pipe(Layer.provide(Layer.succeed(ProcessExecutor, {
      run: () => Effect.sync(() => { calls++; throw new Error("Installer must reject corruption before subprocess execution") }),
    })))))
  expect(result._tag).toBe("Left")
  if (result._tag === "Left") expect(result.left.message).toContain("changed after download")
  expect(calls).toBe(0)
})).pipe(Effect.provide(BunContext.layer))))

test("corruption scenario changes a private copy and requires an integrity rejection", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-corruption-scenario-" })
  const path = join(root, "Magnitude.dmg")
  yield* fs.writeFile(path, payload)
  const candidate = Candidate.make({ artifact: manifest.artifacts[0], version: manifest.version, target, path })
  for (const mode of ["integrity", "unrelated", "accepted"] as const) {
    let corruptPath = "", removed = 0
    const result = yield* rejectCorruptInstaller(candidate).pipe(Effect.provideService(Installer, {
      install: changed => Effect.gen(function* () {
        corruptPath = changed.path
        expect(changed.path).not.toBe(path)
        const bytes = yield* fs.readFile(changed.path).pipe(Effect.orDie)
        expect(bytes.length).toBe(payload.length)
        expect(sha256(bytes)).not.toBe(candidate.artifact.sha256)
        if (mode !== "accepted") return yield* new AssertionFailure({ message: mode === "integrity" ? "Installer changed after download; refusing installation" : "An unrelated install failure" })
        return { candidate: changed, root, executable: "fixture", cli: "fixture", packageVersion: manifest.version }
      }),
      uninstall: () => Effect.sync(() => { removed++ }),
    }), Effect.either)
    expect(result._tag).toBe(mode === "integrity" ? "Right" : "Left")
    expect(removed).toBe(mode === "accepted" ? 1 : 0)
    expect(yield* fs.exists(corruptPath)).toBe(false)
    expect(Buffer.from(yield* fs.readFile(path))).toEqual(Buffer.from(payload))
  }
})).pipe(Effect.provide(BunContext.layer))))
