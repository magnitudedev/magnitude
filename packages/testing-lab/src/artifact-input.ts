import { FileSystem } from "@effect/platform"
import { decodeReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema, Stream } from "effect"
import { dirname, join, resolve } from "node:path"
import { ArtifactStore, fileArtifactStore } from "./artifact-store"
import { Digest, InfrastructureFailure, InvalidInput } from "./domain"
import { ArtifactInput } from "./inputs"
import { sha256 } from "./snapshot"

/** Freeze and verify every declared application artifact before contacting the coordinator. */
export const snapshotArtifacts = (manifestPath: string, objects: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = resolve(manifestPath)
  const release = yield* fs.readFile(path).pipe(Effect.flatMap(decodeReleaseManifest))
  const digests = yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    const digests: Digest[] = []
    for (const artifact of release.artifacts) {
      if (!/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(artifact.filename)) {
        return yield* new InvalidInput({ message: `Unsafe artifact filename: ${artifact.filename}` })
      }
      const file = join(dirname(path), artifact.filename)
      if ((yield* fs.stat(file)).type !== "File") return yield* new InvalidInput({ message: `Artifact is not a regular file: ${artifact.filename}` })
      const digest = Digest.make(artifact.sha256)
      let bytes = 0
      yield* store.put(digest, fs.stream(file).pipe(
        Stream.mapError(() => new InfrastructureFailure({ operation: "artifact-input", message: `Cannot read ${artifact.filename}` })),
        Stream.tap(chunk => Effect.sync(() => { bytes += chunk.byteLength })),
      ))
      if (bytes !== artifact.bytes) return yield* new InvalidInput({ message: `Artifact length does not match manifest: ${artifact.filename}` })
      digests.push(digest)
    }
    return [...new Set(digests)]
  }).pipe(Effect.provide(fileArtifactStore(objects)))
  const json = yield* Schema.encode(Schema.parseJson(ArtifactInput))({ schemaVersion: 1, kind: "artifacts", release, updateAcceptance: Option.none() })
  return { digest: sha256(json), digests, json }
})
