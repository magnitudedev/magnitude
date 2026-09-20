import { FileSystem } from "@effect/platform"
import { isNewerVersion } from "@magnitudedev/release"
import { ReleaseArtifactSchema, ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore, downloadObject } from "./artifact-store"
import { Candidate, prepareCandidate, selectInstaller } from "./candidate"
import { AssertionFailure, Digest, Target } from "./domain"
import { ArtifactInput } from "./inputs"

export const UpdatePair = Schema.Struct({ previous: Candidate, previousRelease: ReleaseManifestSchema, candidate: Candidate,
  update: Schema.Struct({ artifact: ReleaseArtifactSchema, path: Schema.NonEmptyString }) })
const fail = (message: string) => new AssertionFailure({ message })

/** Prepare exact package bytes before any native installation is changed. Publisher trust is checked by the app. */
export const prepareUpdatePair = (baseline: Digest, candidate: typeof ReleaseManifestSchema.Type, target: Target, directory: string) => Effect.gen(function* () {
  const objects = yield* ArtifactStore
  let length = 0
  const chunks = yield* objects.get(baseline).pipe(Stream.tap(chunk => Effect.gen(function* () {
    length += chunk.byteLength
    if (length > 16 * 1024 * 1024) return yield* fail("Previous-release manifest exceeds 16 MiB")
  })), Stream.runCollect)
  const previous = yield* Schema.decodeUnknown(Schema.parseJson(ArtifactInput))(Buffer.concat(Array.from(chunks)).toString("utf8"))
  return yield* prepareReleaseUpdatePair(previous.release, candidate, target, directory)
})

/** Both manifests are admitted before this function materializes any package. */
export const prepareReleaseUpdatePair = (previous: typeof ReleaseManifestSchema.Type, candidate: typeof ReleaseManifestSchema.Type, target: Target, directory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (!isNewerVersion(candidate.version, previous.version)) return yield* fail("Update candidate must be newer than the previous installed version")
  const previousInstaller = yield* selectInstaller(previous, target)
  const candidateInstaller = yield* selectInstaller(candidate, target)
  if (previousInstaller.sha256 === candidateInstaller.sha256) return yield* fail("Different installed versions cannot use identical installer bytes")
  const archives = target.os === "macos"
    ? candidate.artifacts.filter(artifact => artifact.kind === "desktop" && Option.contains(artifact.host, target.artifactHost) && artifact.filename.endsWith(".zip"))
    : [candidateInstaller]
  if (archives.length !== 1) return yield* fail("Expected exactly one update archive for the selected native target")
  const archive = archives[0]!
  if (!/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(archive.filename)) return yield* fail("Update archive filename must be a safe basename")
  const oldPackage = yield* prepareCandidate(previous, target, join(directory, "previous"))
  const newPackage = yield* prepareCandidate(candidate, target, join(directory, "candidate"))
  // Linux and Windows update using their native installer; macOS updates from its distinct ZIP.
  let path = newPackage.path
  if (target.os === "macos") {
    yield* fs.makeDirectory(join(directory, "update"), { recursive: true, mode: 0o700 })
    path = join(directory, "update", archive.filename)
    yield* downloadObject(Digest.make(archive.sha256), path)
  }
  if (Number((yield* fs.stat(path)).size) !== archive.bytes) return yield* fail("Update archive length differs from the admitted manifest")
  return UpdatePair.make({ previous: oldPackage, previousRelease: previous, candidate: newPackage, update: { artifact: archive, path } })
})
