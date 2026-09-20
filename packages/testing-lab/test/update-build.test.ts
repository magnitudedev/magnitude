import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { ArtifactInput, artifactObjects } from "../src/inputs"
import { sha256 } from "../src/snapshot"
import { buildUpdateAcceptance } from "../src/update-build"
import { ProcessExecutorLive } from "../src/process"
import { readManifest } from "../src/build-output"
import { prepareUpdateConsumer } from "../src/update-consumer"
import { targets } from "../src/catalog"
import { UpdateFixtureAuthority } from "../src/update-fixture"

const release = (version: string, content: string, sourceCommit = "a".repeat(40)) => ({ schemaVersion: 2, version, sourceCommit,
  acnRevision: releasePlan.revision, rpc: releasePlan.rpc, plugins: [], tag: `@magnitudedev/cli@${version}`,
  artifacts: [{ id: "fixture", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.deb", sha256: sha256(content), bytes: Buffer.byteLength(content) }],
})
for (const mode of ["success", "wrong-source", "wrong-version"] as const) test(`update producer delivers a distinct admitted fixture graph: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-build-" })
  const input = yield* Schema.decodeUnknown(ArtifactInput)({ schemaVersion: 1, kind: "artifacts", release: release("1.2.3", "normal package") })
  const source = sha256("unpublished source"), phases: string[] = []
  const objectsDirectory = join(root, "objects")
  const invoke = (phase: string, args: readonly string[], env: Readonly<Record<string, string>>) => Effect.gen(function* () {
    phases.push(phase)
    expect(args).toEqual(["packages/testing-lab/scripts/build-update-candidate.ts"])
    expect(env.MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG).toBeTruthy()
    const directory = join(env.MAGNITUDE_ACCEPTANCE_OUTPUT!, "artifacts")
    yield* fs.makeDirectory(directory, { recursive: true })
    const content = `synthetic package for ${phase}`
    yield* fs.writeFileString(join(directory, "Magnitude.deb"), content)
    const value = release(mode === "wrong-version" ? "9.9.9" : env.MAGNITUDE_ACCEPTANCE_VERSION!, content, (mode === "wrong-source" ? "b" : "a").repeat(40))
    yield* fs.writeFileString(join(directory, "release-manifest.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))(value))
  }).pipe(Effect.orDie)
  yield* Effect.gen(function* () {
    const result = yield* buildUpdateAcceptance(source, input, root, objectsDirectory, invoke).pipe(Effect.either)
    expect(result._tag).toBe(mode === "success" ? "Right" : "Left")
    if (result._tag === "Left") { expect(phases).toEqual(["update-previous"]); return }
    const pair = result.right
    expect(phases).toEqual(["update-previous", "update-candidate"])
    expect(pair.previous.version).toBe("1.2.3")
    expect(pair.candidate.version).toBe("1.2.4")
    expect(pair.sourceDigest).toBe(source)
    expect(input.release.version).toBe("1.2.3")
    const combined = ArtifactInput.make({ ...input, updateAcceptance: Option.some(pair) })
    expect(yield* Schema.encode(Schema.parseJson(ArtifactInput))(combined)).not.toContain("PRIVATE KEY")
    const authority = yield* readManifest(pair.authority.sha256, UpdateFixtureAuthority)
    expect(Buffer.byteLength(yield* Schema.encode(Schema.parseJson(UpdateFixtureAuthority))(authority))).toBe(pair.authority.bytes)
    const target = targets.find(value => value.id === "ubuntu-24.04-x64-cpu-intel")!
    for (const changed of [
      { ...pair, authority: { ...pair.authority, bytes: pair.authority.bytes - 1 } },
      { ...pair, authority: { ...pair.authority, bytes: pair.authority.bytes + 1 } },
      { ...pair, configuration: { ...pair.configuration, keyId: "another publisher" } },
    ]) {
      const rejected = yield* Effect.scoped(prepareUpdateConsumer(changed, target, join(root, "rejected"))).pipe(Effect.either)
      expect(rejected._tag).toBe("Left")
    }
    yield* Effect.scoped(Effect.gen(function* () {
      const consumer = yield* prepareUpdateConsumer(pair, target, join(root, "consumer"))
      expect(consumer.fixture.configuration).toEqual(pair.configuration)
      expect(yield* fs.readFileString(consumer.pair.previous.path)).toBe("synthetic package for update-previous")
      expect(yield* fs.readFileString(consumer.pair.update.path)).toBe("synthetic package for update-candidate")
    }))
    const objects = yield* ArtifactStore
    for (const item of artifactObjects(combined).filter(item => item.sha256 !== input.release.artifacts[0].sha256)) {
      const bytes = Buffer.concat(Array.from(yield* objects.get(item.sha256).pipe(Stream.runCollect)))
      expect(bytes.length).toBe(item.bytes)
      expect(sha256(bytes)).toBe(item.sha256)
    }
  }).pipe(Effect.provide(fileArtifactStore(objectsDirectory)))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 20_000)
