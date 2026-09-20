import { FetchHttpClient, FileSystem, HttpClient, HttpClientRequest } from "@effect/platform"
import { Clock, Effect, Layer, Option, Redacted, Schema, Stream, SynchronizedRef } from "effect"
import { join } from "node:path"
import { ArtifactStore, verifiedArtifactContent } from "../artifact-store"
import { Digest, InfrastructureFailure } from "../domain"
import { checkedCommand, ProcessExecutor } from "../process"

export const AzureArtifactConfig = Schema.Struct({ executable: Schema.NonEmptyString, subscription: Schema.UUID,
  account: Schema.String.pipe(Schema.pattern(/^[a-z0-9]{3,24}$/)),
  container: Schema.String.pipe(Schema.pattern(/^[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]$/)),
  maxBytes: Schema.Int.pipe(Schema.between(1, 4 * 1024 ** 3)) })
const failure = (message: string) => new InfrastructureFailure({ operation: "azure-artifacts", message })
const Credential = Schema.Struct({ accessToken: Schema.Redacted(Schema.NonEmptyString), expires_on: Schema.Union(Schema.Int, Schema.NumberFromString) })

/** One renewable Entra credential and shared HTTP client, rather than Azure CLI processes per object. */
export const azureArtifactStore = (config: typeof AzureArtifactConfig.Type) => Layer.effect(ArtifactStore, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const http = yield* HttpClient.HttpClient
  const cached = yield* SynchronizedRef.make(Option.none<typeof Credential.Type>())
  const credential = SynchronizedRef.modifyEffect(cached, current => Effect.gen(function* () {
    const now = yield* Clock.currentTimeMillis
    if (Option.isSome(current) && current.value.expires_on * 1000 > now + 120_000) return [current.value.accessToken, current] as const
    const reply = yield* checkedCommand(config.executable, ["account", "get-access-token", "--subscription", config.subscription,
      "--resource", "https://storage.azure.com/", "--only-show-errors", "--output", "json"], { timeoutMs: 30_000, maxOutputBytes: 64 * 1024 }).pipe(
      Effect.provideService(ProcessExecutor, executor), Effect.mapError(() => failure("Cannot acquire Azure storage credential")))
    const next = yield* Schema.decodeUnknown(Schema.parseJson(Credential))(reply.stdout).pipe(Effect.mapError(() => failure("Malformed Azure storage credential")))
    if (next.expires_on * 1000 <= now + 120_000) return yield* failure("Azure storage credential expires too soon")
    return [next.accessToken, Option.some(next)] as const
  }))
  const url = (digest: Digest) => `https://${config.account}.blob.core.windows.net/${config.container}/${digest}`
  const send = (request: HttpClientRequest.HttpClientRequest) => Effect.gen(function* () {
    const token = yield* credential
    return yield* http.execute(request.pipe(HttpClientRequest.setHeaders({
      authorization: `Bearer ${Redacted.value(token)}`, "x-ms-version": "2023-11-03", "x-ms-date": new Date().toUTCString(),
    }))).pipe(Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
      Effect.mapError(() => failure("Azure blob HTTP request failed")))
  }).pipe(Effect.timeoutFail({ duration: "10 minutes", onTimeout: () => failure("Azure blob HTTP request timed out") }))
  const inspect = (digest: Digest) => Effect.gen(function* () {
    const response = yield* send(HttpClientRequest.head(url(digest)))
    if (response.status === 404) return Option.none()
    if (response.status !== 200) return yield* failure(`Azure blob metadata returned HTTP ${response.status}`)
    const length = response.headers["content-length"]
    const bytes = length && /^\d+$/.test(length) ? Number(length) : NaN
    const etag = response.headers.etag
    if (!Number.isSafeInteger(bytes) || bytes < 0 || !etag) return yield* failure("Invalid Azure blob metadata")
    return Option.some({ bytes, etag })
  })
  return {
    put: (digest, content) => Effect.scoped(Effect.gen(function* () {
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-blob-upload-" })
      const file = join(directory, digest)
      yield* Stream.run(verifiedArtifactContent(digest, content, config.maxBytes), fs.sink(file, { flag: "wx", mode: 0o600 }))
      const bytes = Number((yield* fs.stat(file)).size)
      const existing = yield* inspect(digest)
      if (Option.isSome(existing)) {
        if (existing.value.bytes !== bytes) return yield* failure("Existing Azure object has a different length")
        return
      }
      const request = HttpClientRequest.put(url(digest)).pipe(HttpClientRequest.setHeaders({ "x-ms-blob-type": "BlockBlob", "if-none-match": "*" }))
      const body = bytes === 0 ? HttpClientRequest.bodyUint8Array(request, new Uint8Array(0), "application/octet-stream")
        : HttpClientRequest.bodyStream(request, fs.stream(file), { contentType: "application/octet-stream", contentLength: bytes })
      // Effect's body helper omits Content-Length when it is zero; Azure requires the header.
      const response = yield* send(body.pipe(HttpClientRequest.setHeader("content-length", String(bytes))))
      if (response.status === 201) return
      // A concurrent verified writer may have won the immutable content address.
      if (response.status === 412) {
        const winner = yield* inspect(digest)
        if (Option.isSome(winner) && winner.value.bytes === bytes) return
      }
      return yield* failure(`Azure blob upload returned HTTP ${response.status}`)
    })).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure("Cannot stage cloud artifact"))),
    get: digest => Stream.unwrap(Effect.gen(function* () {
      const metadata = yield* inspect(digest)
      if (Option.isNone(metadata)) return yield* failure("Azure blob does not exist")
      if (metadata.value.bytes > config.maxBytes) return yield* failure("Cloud artifact exceeds the configured byte limit")
      const response = yield* send(HttpClientRequest.get(url(digest)).pipe(HttpClientRequest.setHeader("if-match", metadata.value.etag)))
      if (response.status !== 200) return yield* failure(`Azure blob download returned HTTP ${response.status}`)
      if (response.headers.etag !== metadata.value.etag || response.headers["content-length"] !== String(metadata.value.bytes)) return yield* failure("Azure blob changed during download")
      return verifiedArtifactContent(digest, response.stream.pipe(Stream.mapError(() => failure("Azure blob download interrupted"))), metadata.value.bytes).pipe(
        Stream.timeoutFail(() => failure("Azure blob download stalled"), "2 minutes"),
      )
    })),
    exists: digest => inspect(digest).pipe(Effect.map(Option.isSome)),
  } satisfies ArtifactStore
}))
