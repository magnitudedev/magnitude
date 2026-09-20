import { Effect, Schema, Stream } from "effect"
import { ArtifactStore } from "./artifact-store"
import { Backend, Digest, InfrastructureFailure, Target } from "./domain"
import { ArtifactInput } from "./inputs"
import { sha256, SourceManifest } from "./snapshot"

/** Package identity is independent of the machine that subsequently installs it. */
export const BuildOutput = Schema.Struct({ sourceDigest: Digest, sourceCommit: SourceManifest.fields.commit,
  artifactDigest: Digest, artifactHost: Target.fields.artifactHost, backend: Backend })
export type BuildOutput = typeof BuildOutput.Type
export const readManifest = <A, I>(digest: Digest, schema: Schema.Schema<A, I>) => Effect.gen(function* () {
  const store = yield* ArtifactStore
  let size = 0
  const chunks = yield* store.get(digest).pipe(Stream.tap(chunk => Effect.gen(function* () {
    size += chunk.byteLength
    if (size > 16 * 1024 * 1024) return yield* new InfrastructureFailure({ operation: "build-output", message: "Manifest exceeds 16 MiB" })
  })), Stream.runCollect)
  const bytes = Buffer.concat(Array.from(chunks))
  if (sha256(bytes) !== digest) return yield* new InfrastructureFailure({ operation: "build-output", message: "Manifest content address mismatch" })
  return yield* Schema.decodeUnknown(Schema.parseJson(schema))(bytes.toString("utf8")).pipe(
    Effect.mapError(() => new InfrastructureFailure({ operation: "build-output", message: "Malformed manifest" })))
})
export const outputObjects = (output: BuildOutput) => Effect.gen(function* () {
  const input = yield* readManifest(output.artifactDigest, ArtifactInput)
  const wire = yield* Schema.encode(Schema.parseJson(ArtifactInput))(input).pipe(
    Effect.mapError(() => new InfrastructureFailure({ operation: "build-output", message: "Cannot encode output manifest" })))
  return [{ digest: output.artifactDigest, bytes: Buffer.byteLength(wire) },
    ...input.release.artifacts.map(item => ({ digest: Digest.make(item.sha256), bytes: item.bytes }))]
})
