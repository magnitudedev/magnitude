import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema, Stream } from "effect"
import { dirname, join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { NodeArchiveExtractor } from "../../release/src/archive"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { targets } from "../src/catalog"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { admittedRuntimeComposition } from "../src/runtime-composition"
import { sha256 } from "../src/snapshot"

test("composes verified archives, keeps CPU selection explicit and rejects identity, path and byte conflicts", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-composition-fixture-" })
  const objects = join(temporary, "objects")
  yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    const archive = (name: string, files: Readonly<Record<string, string>>) => Effect.gen(function* () {
      const root = join(temporary, name)
      for (const [path, contents] of Object.entries(files)) {
        yield* fs.makeDirectory(dirname(join(root, path)), { recursive: true })
        yield* fs.writeFileString(join(root, path), contents)
      }
      const output = join(temporary, `${name}.tar.gz`)
      yield* checkedCommand("tar", ["-czf", output, "-C", root, ...Object.keys(files)], { env: { COPYFILE_DISABLE: "1" } })
      const bytes = yield* fs.readFile(output), digest = sha256(bytes)
      yield* store.put(digest, Stream.make(bytes))
      return { filename: `${name}.tar.gz`, bytes: bytes.length, sha256: digest }
    })
    const base = yield* archive("base", { "bin/magnitude-inference": "binary fixture", "catalog/model-planner-inputs.bundle": "catalog", "backends/cpu.dylib": "cpu" })
    const metal = yield* archive("metal", { "backends/metal.dylib": "metal", "runtime/owned.dylib": "runtime" })
    const duplicate = yield* archive("duplicate", { "backends/metal.dylib": "metal", "backends/cpu.dylib": "collision" })
    const release = yield* Schema.decodeUnknown(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
      rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [
        { ...base, id: "base", kind: "icn-base", host: "darwin-arm64", backend: "cpu", nativeBuild: "native-fixture", backendModuleAbi: "abi" },
        { ...metal, id: "metal", kind: "icn-backend", host: "darwin-arm64", backend: "metal", nativeBuild: "native-fixture", backendModuleAbi: "abi", requiredBaseId: "base", compatibility: { kind: "metal" } },
      ] })
    const target = targets.find(target => target.id === "macos-15-arm64-metal-apple-silicon")!
    let composed = ""
    yield* Effect.scoped(Effect.gen(function* () {
      const result = yield* admittedRuntimeComposition(release, target)
      composed = result.root
      expect(result.artifacts).toEqual(["base", "metal"])
      expect(yield* fs.readFileString(join(result.root, "runtime", "owned.dylib"))).toBe("runtime")
      expect((yield* fs.readDirectory(join(result.root, "backends"))).sort()).toEqual(["cpu.dylib", "metal.dylib"])
      const cpu = yield* admittedRuntimeComposition(release, { ...target, backend: "cpu" })
      expect(cpu.artifacts).toEqual(["base"])
      expect(yield* fs.readDirectory(join(cpu.root, "backends"))).toEqual(["cpu.dylib"])
    }))
    expect(yield* fs.exists(composed)).toBe(false)
    const baseArtifact = release.artifacts[0]!, pack = release.artifacts[1]!
    for (const changed of [{ ...pack, nativeBuild: Option.some("different") }, { ...pack, backendModuleAbi: Option.some("different") },
      { ...pack, ...duplicate }, { ...pack, bytes: pack.bytes + 1 }]) {
      expect((yield* Effect.scoped(admittedRuntimeComposition({ ...release, artifacts: [baseArtifact, changed] }, target)).pipe(Effect.either))._tag).toBe("Left")
    }
    expect((yield* Effect.scoped(admittedRuntimeComposition({ ...release, artifacts: [baseArtifact] }, target)).pipe(Effect.either))._tag).toBe("Left")
    yield* fs.writeFileString(join(objects, base.sha256), "corrupt CAS bytes")
    expect((yield* Effect.scoped(admittedRuntimeComposition(release, target)).pipe(Effect.either))._tag).toBe("Left")
  }).pipe(Effect.provide(fileArtifactStore(objects)))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, NodeArchiveExtractor]))))
