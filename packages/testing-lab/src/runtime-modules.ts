import { FileSystem } from "@effect/platform"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { ArchiveExtractor } from "../../release/src/archive"
import { downloadObject } from "./artifact-store"
import { AssertionFailure, Digest, InfrastructureFailure, Target } from "./domain"
import { LoadedBackendModule, NativeExecution } from "./execution-telemetry"

/** Expected module identities come from verified admitted archives, never a developer install. */
export const admittedRuntimeModules = (release: typeof ReleaseManifestSchema.Type, host: Target["artifactHost"]) => Effect.scoped(Effect.gen(function* () {
  yield* validateReleaseManifest(release)
  const fs = yield* FileSystem.FileSystem
  const extractor = yield* ArchiveExtractor
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "ml-module-identity-" })
  const artifacts = release.artifacts.filter(artifact => Option.contains(artifact.host, host)
    && (artifact.kind === "icn-base" || artifact.kind === "icn-backend"))
  if (artifacts.filter(artifact => artifact.kind === "icn-base").length !== 1) {
    return yield* new AssertionFailure({ message: "Module verification requires an admitted native base for this host" })
  }
  const modules: (typeof LoadedBackendModule.Type)[] = []
  for (const [index, artifact] of artifacts.entries()) {
    const archive = join(root, `${index}.tar.gz`), destination = join(root, String(index))
    yield* downloadObject(Digest.make(artifact.sha256), archive)
    if (Number((yield* fs.stat(archive)).size) !== artifact.bytes) {
      return yield* new AssertionFailure({ message: "Native archive length differs from admitted metadata" })
    }
    yield* extractor.extract(archive, destination, artifact, Option.none())
    const directory = join(destination, "backends")
    for (const name of yield* fs.readDirectory(directory)) {
      const path = join(directory, name)
      const stat = yield* fs.stat(path)
      if (stat.type !== "File" || modules.some(module => module.name === name)) {
        return yield* new AssertionFailure({ message: "Native archives contain an ambiguous backend module" })
      }
      const hash = createHash("sha256")
      yield* fs.stream(path).pipe(Stream.runForEach(bytes => Effect.sync(() => { hash.update(bytes) })))
      modules.push(LoadedBackendModule.make({ name, bytes: Number(stat.size), sha256: Digest.make(hash.digest("hex")) }))
    }
  }
  return modules
})).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error
  : new InfrastructureFailure({ operation: "runtime-modules", message: error.message })))

/** Module identity is necessary but does not by itself establish GPU allocation or device identity. */
export const attestRuntimeModules = (execution: NativeExecution, expected: readonly (typeof LoadedBackendModule.Type)[]) => Effect.gen(function* () {
  if (Option.isNone(execution.modules) || execution.modules.value.length === 0) {
    return yield* new AssertionFailure({ message: "Generation has no loaded backend module evidence" })
  }
  const seen = new Set<string>()
  for (const observed of execution.modules.value) {
    const matches = expected.filter(module => module.name === observed.name)
    if (seen.has(observed.name) || matches.length !== 1 || matches[0]!.sha256 !== observed.sha256 || matches[0]!.bytes !== observed.bytes) {
      return yield* new AssertionFailure({ message: "Generation used a backend module that does not uniquely match the admitted runtime" })
    }
    seen.add(observed.name)
  }
})
