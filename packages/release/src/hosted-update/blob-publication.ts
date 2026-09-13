import { FileSystem, HttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { createReadStream } from "node:fs"
import { BlobNotFoundError, head, put } from "@vercel/blob"
import type { UpdateManifest } from "./manifest"

export class ArtifactPublicationFailed extends Schema.TaggedError<ArtifactPublicationFailed>()("ArtifactPublicationFailed", {
  stage: Schema.Literal("local-verification", "upload", "remote-verification"),
  message: Schema.String,
}) {}

/** Immutable storage publication. Both new uploads and idempotent repeats require a full remote digest check. */
export const publishArtifactBytes = (options: {
  readonly file: string
  readonly manifest: UpdateManifest
  readonly storageOrigin: string
  readonly token: string
}) => Effect.gen(function* () {
  const artifact = options.manifest.artifact
  const fs = yield* FileSystem.FileSystem
  const stat = yield* fs.stat(options.file).pipe(Effect.mapError(() => new ArtifactPublicationFailed({ stage: "local-verification", message: "Could not inspect the local artifact" })))
  if (stat.size !== BigInt(artifact.bytes)) return yield* new ArtifactPublicationFailed({ stage: "local-verification", message: "Local artifact size does not match its descriptor" })
  const hash = yield* fs.stream(options.file).pipe(Stream.runFold(createHash("sha256"), (hash, chunk) => hash.update(chunk)),
    Effect.mapError(() => new ArtifactPublicationFailed({ stage: "local-verification", message: "Could not read the local artifact" })))
  if (hash.digest("hex") !== artifact.sha256) return yield* new ArtifactPublicationFailed({ stage: "local-verification", message: "Local artifact digest does not match its descriptor" })
  const url = yield* Effect.try({ try: () => {
    const url = new URL(artifact.path, options.storageOrigin + "/")
    if (url.protocol !== "https:" || url.origin !== options.storageOrigin) throw new Error("Invalid storage origin")
    return url
  }, catch: () => new ArtifactPublicationFailed({ stage: "upload", message: "Invalid artifact storage destination" }) })
  // Intentional Blob SDK boundary. Its request signals and file stream belong to this Effect.
  yield* Effect.tryPromise({ try: async signal => {
    try {
      await head(url.href, { token: options.token, abortSignal: signal })
    } catch (error) {
      if (!(error instanceof BlobNotFoundError)) throw error
      const stream = createReadStream(options.file)
      try {
        const uploaded = await put(artifact.path, stream, { access: "public", token: options.token, addRandomSuffix: false,
          allowOverwrite: false, multipart: true, cacheControlMaxAge: 31536000, abortSignal: signal })
        if (uploaded.url !== url.href) throw new Error("Unexpected artifact destination")
      } finally { stream.destroy() }
    }
  }, catch: () => new ArtifactPublicationFailed({ stage: "upload", message: "Could not publish the immutable artifact" }) })
  const http = yield* HttpClient.HttpClient
  const response = yield* http.execute(HttpClientRequest.get(url.href)).pipe(Effect.mapError(() => new ArtifactPublicationFailed({ stage: "remote-verification", message: "Remote artifact request failed" })))
  if (response.status !== 200) return yield* new ArtifactPublicationFailed({ stage: "remote-verification", message: `Remote artifact returned HTTP ${response.status}` })
  const received = yield* response.stream.pipe(Stream.runFoldEffect({ bytes: 0, digest: createHash("sha256") }, (state, chunk) => {
    const bytes = state.bytes + chunk.byteLength
    return bytes > artifact.bytes ? new ArtifactPublicationFailed({ stage: "remote-verification", message: `Remote artifact exceeded its expected ${artifact.bytes} bytes` })
      : Effect.succeed({ bytes, digest: state.digest.update(chunk) })
  }), Effect.mapError(error => error instanceof ArtifactPublicationFailed ? error : new ArtifactPublicationFailed({ stage: "remote-verification", message: "Remote artifact stream failed before completion" })))
  if (received.bytes !== artifact.bytes) return yield* new ArtifactPublicationFailed({ stage: "remote-verification", message: `Remote artifact contained ${received.bytes} bytes; expected ${artifact.bytes}` })
  if (received.digest.digest("hex") !== artifact.sha256) return yield* new ArtifactPublicationFailed({ stage: "remote-verification", message: "Remote artifact digest does not match the verified local artifact" })
  return url.href
}).pipe(Effect.timeoutFail({ duration: "30 minutes", onTimeout: () => new ArtifactPublicationFailed({ stage: "upload", message: "Artifact publication exceeded its deadline" }) }))
