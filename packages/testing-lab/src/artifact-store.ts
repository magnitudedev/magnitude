import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { Digest, InfrastructureFailure } from "./domain"

export interface ArtifactStore {
  readonly put: (digest: Digest, content: Stream.Stream<Uint8Array, InfrastructureFailure>) => Effect.Effect<void, InfrastructureFailure>
  /** Consumers must drain the stream successfully before accepting the object. */
  readonly get: (digest: Digest) => Stream.Stream<Uint8Array, InfrastructureFailure>
  readonly exists: (digest: Digest) => Effect.Effect<boolean, InfrastructureFailure>
}
export const ArtifactStore = Context.GenericTag<ArtifactStore>("@magnitudedev/testing-lab/ArtifactStore")
const failure = (message: string) => new InfrastructureFailure({ operation: "artifact-store", message })
const verified = (digest: Digest, content: Stream.Stream<Uint8Array, InfrastructureFailure>, maxBytes: number) => Stream.unwrap(Effect.sync(() => {
  const hash = createHash("sha256")
  let bytes = 0
  return content.pipe(Stream.filter(chunk => chunk.byteLength > 0), Stream.tap(chunk => Effect.gen(function* () {
    bytes += chunk.byteLength
    if (bytes > maxBytes) return yield* failure("Object exceeds the configured byte limit")
    hash.update(chunk)
  })), Stream.concat(Stream.drain(Stream.fromEffect(Effect.gen(function* () {
    if (hash.digest("hex") !== digest) return yield* failure("Object SHA-256 does not match its content address")
  })))))
}))
export const fileArtifactStore = (root: string, maxBytes = 4 * 1024 ** 3) => Layer.effect(ArtifactStore, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 }).pipe(Effect.mapError(() => failure("Cannot create object storage")))
  return {
    put: (digest, content) => Effect.scoped(Effect.gen(function* () {
      const temporary = join(root, `.upload-${crypto.randomUUID()}`)
      yield* Effect.addFinalizer(() => fs.remove(temporary, { force: true }).pipe(Effect.orDie))
      yield* Stream.run(verified(digest, content, maxBytes), fs.sink(temporary, { flag: "wx", mode: 0o600 }))
      yield* fs.rename(temporary, join(root, digest))
    })).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure("Cannot persist object"))),
    get: digest => verified(digest, fs.stream(join(root, digest)).pipe(Stream.mapError(() => failure("Cannot read object"))), maxBytes),
    exists: digest => fs.exists(join(root, digest)).pipe(Effect.mapError(() => failure("Cannot inspect object"))),
  } satisfies ArtifactStore
}))

/** Temp-file publication ensures a corrupt or interrupted download never becomes an accepted file. */
export const downloadObject = (digest: Digest, destination: string) => Effect.gen(function* () {
  const store = yield* ArtifactStore
  const fs = yield* FileSystem.FileSystem
  return yield* Effect.scoped(Effect.gen(function* () {
    const temporary = `${destination}.download-${crypto.randomUUID()}`
    yield* Effect.addFinalizer(() => fs.remove(temporary, { force: true }).pipe(Effect.orDie))
    yield* Stream.run(store.get(digest), fs.sink(temporary, { flag: "wx", mode: 0o600 }))
    yield* fs.rename(temporary, destination)
  })).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure("Cannot publish downloaded object")))
})
