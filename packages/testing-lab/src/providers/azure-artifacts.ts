import { FileSystem } from "@effect/platform"
import { Effect, Layer, Schema, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore, verifiedArtifactContent } from "../artifact-store"
import { InfrastructureFailure } from "../domain"
import { checkedCommand, ProcessExecutor } from "../process"

export const AzureArtifactConfig = Schema.Struct({ executable: Schema.NonEmptyString, subscription: Schema.UUID,
  account: Schema.String.pipe(Schema.pattern(/^[a-z0-9]{3,24}$/)),
  container: Schema.String.pipe(Schema.pattern(/^[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]$/)),
  maxBytes: Schema.Int.pipe(Schema.between(1, 4 * 1024 ** 3)) })
const failure = (message: string) => new InfrastructureFailure({ operation: "azure-artifacts", message })
/** Coordinator identity uses Entra data-plane access; no storage account key enters the worker. */
export const azureArtifactStore = (config: typeof AzureArtifactConfig.Type) => Layer.effect(ArtifactStore, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const run = (args: readonly string[]) => checkedCommand(config.executable, ["storage", "blob", ...args, "--subscription", config.subscription,
    "--account-name", config.account, "--container-name", config.container, "--auth-mode", "login", "--only-show-errors", "--output", "json"],
    { timeoutMs: 900_000, maxOutputBytes: 1024 * 1024 }).pipe(Effect.provideService(ProcessExecutor, executor), Effect.mapError(() => failure("Authenticated Azure blob operation failed")))
  const exists = (digest: string) => run(["exists", "--name", digest]).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ exists: Schema.Boolean })))(result.stdout)),
    Effect.map(result => result.exists), Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure(error.message)))
  return {
    put: (digest, content) => Effect.scoped(Effect.gen(function* () {
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-blob-upload-" })
      const file = join(directory, digest)
      yield* Stream.run(verifiedArtifactContent(digest, content, config.maxBytes), fs.sink(file, { flag: "wx", mode: 0o600 }))
      // Publish once. A concurrent verified writer may win the same content address.
      if (!(yield* exists(digest))) yield* run(["upload", "--name", digest, "--file", file, "--overwrite", "false", "--type", "block", "--no-progress"]).pipe(
        Effect.catchAll(error => exists(digest).pipe(Effect.flatMap(present => present ? Effect.void : Effect.fail(error)))))
    })).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure(error.message))),
    get: digest => Stream.unwrapScoped(Effect.gen(function* () {
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "lab-blob-download-" })
      const file = join(directory, digest)
      // Check the cloud length before downloading; verification still bounds and hashes the stream.
      const metadata = yield* run(["show", "--name", digest]).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ properties: Schema.Struct({ contentLength: Schema.Int.pipe(Schema.nonNegative()), etag: Schema.NonEmptyString }) })))(result.stdout)))
      if (metadata.properties.contentLength > config.maxBytes) return yield* failure("Cloud artifact exceeds the configured byte limit")
      yield* run(["download", "--name", digest, "--file", file, "--if-match", metadata.properties.etag, "--no-progress"])
      if (Number((yield* fs.stat(file)).size) !== metadata.properties.contentLength) return yield* failure("Cloud artifact changed length during download")
      return verifiedArtifactContent(digest, fs.stream(file).pipe(Stream.mapError(() => failure("Cannot read staged cloud artifact"))), config.maxBytes)
    }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure(error.message)))),
    exists,
  } satisfies ArtifactStore
}))
