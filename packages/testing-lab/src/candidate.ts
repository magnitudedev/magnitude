import { FileSystem } from "@effect/platform"
import { ReleaseArtifactSchema, ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { downloadObject } from "./artifact-store"
import { AssertionFailure, Digest, InfrastructureFailure, Target } from "./domain"

export const Candidate = Schema.Struct({ artifact: ReleaseArtifactSchema, version: Schema.NonEmptyString, target: Target, path: Schema.NonEmptyString })
export type Candidate = typeof Candidate.Type
export const selectInstaller = (release: typeof ReleaseManifestSchema.Type, target: Target) => Effect.gen(function* () {
  yield* validateReleaseManifest(release).pipe(Effect.mapError(e => new AssertionFailure({ message: e.message })))
  const artifacts = release.artifacts.filter(a => a.kind === "desktop" && Option.contains(a.host, target.artifactHost) && a.filename.endsWith(`.${target.packageFormat}`))
  if (artifacts.length !== 1) return yield* new AssertionFailure({ message: `Expected exactly one ${target.packageFormat} desktop installer for ${target.artifactHost}; found ${artifacts.length}` })
  const artifact = artifacts[0]!
  if (!/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(artifact.filename)) return yield* new AssertionFailure({ message: "Installer filename must be a safe basename" })
  return artifact
})

/** Materialize the content-addressed installer, never a release URL or builder directory. */
export const prepareCandidate = (release: typeof ReleaseManifestSchema.Type, target: Target, directory: string) => Effect.scoped(Effect.gen(function* () {
  const artifact = yield* selectInstaller(release, target)
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(directory, { recursive: true, mode: 0o700 })
  const path = join(directory, artifact.filename)
  const stage = yield* fs.makeTempDirectoryScoped({ directory, prefix: ".candidate-" })
  const staged = join(stage, artifact.filename)
  yield* downloadObject(Digest.make(artifact.sha256), staged)
  if (Number((yield* fs.stat(staged)).size) !== artifact.bytes) return yield* new AssertionFailure({ message: "Downloaded installer length differs from the accepted manifest" })
  yield* fs.rename(staged, path)
  return Candidate.make({ artifact, version: release.version, target, path })
})).pipe(Effect.mapError(error => error._tag === "AssertionFailure" || error._tag === "InfrastructureFailure" ? error
  : new InfrastructureFailure({ operation: "candidate-download", message: error.message })))
