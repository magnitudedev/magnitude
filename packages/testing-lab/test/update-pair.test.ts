import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Array as EffectArray, Effect, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { targets } from "../src/catalog"
import { sha256 } from "../src/snapshot"
import { prepareUpdatePair } from "../src/update-pair"

for (const id of ["macos-15-arm64-metal-apple-silicon", "windows-server-2025-x64-cpu-intel", "ubuntu-24.04-x64-cpu-intel", "fedora-44-x64-cpu-intel"]) {
  test(`prepares exact old/new installer and update bytes for ${id}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const target = targets.find(target => target.id === id)
    if (!target) return yield* Effect.dieMessage(`Missing test target ${id}`)
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-pair-" })
    const make = (version: string) => Schema.decodeUnknownSync(ReleaseManifestSchema)({ schemaVersion: 2, version, acnRevision: 1,
      rpc: releasePlan.rpc, plugins: [], tag: `@magnitudedev/cli@${version}`, sourceCommit: "a".repeat(40), artifacts:
        [target.packageFormat, ...(target.os === "macos" ? ["zip"] : [])].map(format => {
          const bytes = `${version}-${format}`
          return { id: `desktop-${target.artifactHost}-${format}`, kind: "desktop", host: target.artifactHost,
            filename: `Magnitude.${format}`, bytes: bytes.length, sha256: sha256(bytes) }
        }) })
    const previous = make("0.1.2"), candidate = make("0.1.3")
    const baseline = yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 1, kind: "artifacts", release: yield* Schema.encode(ReleaseManifestSchema)(previous) })
    yield* Effect.gen(function* () {
      const objects = yield* ArtifactStore
      yield* objects.put(sha256(baseline), Stream.make(new TextEncoder().encode(baseline)))
      for (const release of [previous, candidate]) for (const artifact of release.artifacts) {
        const content = `${release.version}-${artifact.filename.split(".").at(-1)}`
        yield* objects.put(sha256(content), Stream.make(new TextEncoder().encode(content)))
      }
      expect((yield* prepareUpdatePair(sha256(baseline), candidate, { ...target, artifactHost: "linux-arm64-gnu" }, join(root, "wrong-host")).pipe(Effect.either))._tag).toBe("Left")
      expect((yield* prepareUpdatePair(sha256(baseline), { ...candidate, artifacts: EffectArray.map(candidate.artifacts, a => ({ ...a, bytes: a.bytes + 1 })) }, target, join(root, "wrong-length")).pipe(Effect.either))._tag).toBe("Left")
      const pair = yield* prepareUpdatePair(sha256(baseline), candidate, target, join(root, "prepared"))
      expect(pair.previous.version).toBe("0.1.2")
      expect(pair.candidate.version).toBe("0.1.3")
      expect(yield* fs.readFileString(pair.previous.path)).toBe(`0.1.2-${target.packageFormat}`)
      expect(yield* fs.readFileString(pair.update.path)).toBe(`0.1.3-${target.os === "macos" ? "zip" : target.packageFormat}`)
      expect(pair.update.path === pair.candidate.path).toBe(target.os !== "macos")
      for (const version of ["0.1.2", "0.1.1"]) expect((yield* prepareUpdatePair(sha256(baseline), make(version), target, join(root, version)).pipe(Effect.either))._tag).toBe("Left")
      expect((yield* prepareUpdatePair(sha256(baseline), { ...candidate, artifacts: previous.artifacts }, target, join(root, "relabelled")).pipe(Effect.either))._tag).toBe("Left")
      if (target.os === "macos") {
        expect((yield* prepareUpdatePair(sha256(baseline), { ...candidate, artifacts: [candidate.artifacts[0]] }, target, join(root, "missing-zip")).pipe(Effect.either))._tag).toBe("Left")
        expect((yield* prepareUpdatePair(sha256(baseline), { ...candidate, artifacts: [...candidate.artifacts, { ...candidate.artifacts[1]!, id: "duplicate", filename: "other.zip" }] }, target, join(root, "duplicate-zip")).pipe(Effect.either))._tag).toBe("Left")
      }
      yield* fs.writeFileString(join(root, "objects", pair.update.artifact.sha256), "corrupt bytes")
      expect((yield* prepareUpdatePair(sha256(baseline), candidate, target, join(root, "corrupt")).pipe(Effect.either))._tag).toBe("Left")
    }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
  })).pipe(Effect.provide(BunContext.layer))))
}
