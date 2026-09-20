import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import releasePlan from "../../release/release-plan.json"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { targets } from "../src/catalog"
import { ProcessExecutor } from "../src/process"
import { nativeSourceBuilder, SourceBuilder } from "../src/source-builder"
import { manifestJson, RelativePath, sha256, SourceManifest } from "../src/snapshot"

for (const mode of ["success", "compile-failure", "package-failure", "wrong-provenance", "corrupt-package"] as const) test(`source producer preserves phase and bytes: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-source-builder-" })
  const sourceBytes = new TextEncoder().encode("unpublished local change")
  const source = SourceManifest.make({ schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: [
    { kind: "file", path: RelativePath.make("dirty.txt"), sha256: sha256(sourceBytes), bytes: sourceBytes.byteLength, executable: false },
  ] })
  const digest = sha256(manifestJson(source)), phases: string[] = []
  const target = targets.find(t => t.arch === process.arch && t.os === (process.platform === "darwin" ? "macos" : process.platform === "win32" ? "windows" : "ubuntu"))!
  const objectsRoot = join(root, "objects")
  const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.gen(function* () {
    const phase = spec.env.LAB_BUILD_PHASE!
    phases.push(phase)
    expect(spec.inheritEnv).toBe(false)
    expect(spec.env.LAB_BUILD_SOURCE_DIGEST).toBe(digest)
    expect(spec.env.LAB_BUILD_BACKEND).toBe(target.backend)
    expect(yield* fs.readFileString(join(Option.getOrThrow(spec.cwd), "dirty.txt"))).toBe("unpublished local change")
    expect(spec.env.HOME).toBe(join(root, "build", "home"))
    if (phase === "dependencies") {
      const args = process.platform === "win32"
        ? yield* Schema.decodeUnknown(Schema.parseJson(Schema.Array(Schema.String)))(Buffer.from(spec.args[spec.args.indexOf("-ArgumentsBase64") + 1]!, "base64").toString())
        : spec.args
      expect(args).toEqual(["install", "--frozen-lockfile"])
      if (process.platform === "win32") expect(spec.executable).toBe("pwsh.exe")
    }
    const fails = mode === "compile-failure" && phase === "compile" || mode === "package-failure" && phase === "package"
    if (phase === "package" && !fails) {
      const artifact = new TextEncoder().encode("explicit package fixture")
      const directory = join(spec.env.LAB_BUILD_OUTPUT!, "artifacts")
      yield* fs.makeDirectory(directory, { recursive: true })
      const filename = `Magnitude.${target.packageFormat}`
      yield* fs.writeFile(join(directory, filename), mode === "corrupt-package" ? new TextEncoder().encode("changed fixture") : artifact)
      yield* fs.writeFileString(join(directory, "release-manifest.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ schemaVersion: 2, version: releasePlan.cliVersion,
        acnRevision: releasePlan.revision, rpc: releasePlan.rpc, plugins: [], tag: `@magnitudedev/cli@${releasePlan.cliVersion}`,
        sourceCommit: (mode === "wrong-provenance" ? "b" : "a").repeat(40), artifacts: [{ id: "fixture", kind: "desktop", host: target.artifactHost,
          filename, sha256: sha256(artifact), bytes: artifact.byteLength }] }))
    }
    return { exitCode: fails ? 1 : 0, stdout: `fixture ${phase}`, stderr: fails ? "intentional phase failure" : "" }
  }).pipe(Effect.orDie) })
  const store = fileArtifactStore(objectsRoot)
  yield* Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(sourceBytes), Stream.make(sourceBytes))
    const stages = yield* (yield* SourceBuilder).prepare(source, digest, target, target.backend)
    const compile = yield* stages.compile.pipe(Effect.either)
    expect(compile._tag).toBe(mode === "compile-failure" ? "Left" : "Right")
    const result = yield* stages.package.pipe(Effect.either)
    expect(result._tag).toBe(mode === "success" ? "Right" : "Left")
    expect(phases).toEqual(mode === "compile-failure" ? ["dependencies", "compile"] : ["dependencies", "compile", "package"])
    if (result._tag === "Right") {
      expect(result.right.evidence).toHaveLength(3)
      for (const item of result.right.evidence) expect(yield* objects.exists(item.sha256)).toBe(true)
      yield* stages.package
      expect(phases).toHaveLength(3)
    }
  }).pipe(Effect.provide(Layer.merge(store, nativeSourceBuilder({ root: join(root, "build"), objects: objectsRoot, environment: { PATH: "/usr/bin:/bin" } }).pipe(Layer.provide(Layer.merge(store, executor))))))
})).pipe(Effect.provide(BunContext.layer))))
