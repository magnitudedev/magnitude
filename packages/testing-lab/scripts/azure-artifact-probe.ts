import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { ArtifactStore, downloadObject } from "../src/artifact-store"
import { Digest, InfrastructureFailure } from "../src/domain"
import { azureArtifactStore, AzureArtifactConfig } from "../src/providers/azure-artifacts"
import { ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const file = yield* Config.string("LAB_ARTIFACT_FILE")
  const report = yield* Config.string("LAB_ARTIFACT_REPORT")
  const config = yield* fs.readFileString(yield* Config.string("LAB_AZURE_ARTIFACT_CONFIG")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(AzureArtifactConfig))))
  const hash = createHash("sha256")
  let bytes = 0
  yield* fs.stream(file).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk); bytes += chunk.byteLength })))
  const digest = Digest.make(hash.digest("hex"))
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: "lab-azure-artifact-probe-" })
  yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    yield* store.put(digest, fs.stream(file).pipe(Stream.mapError(error => new InfrastructureFailure({ operation: "probe-upload", message: error.message }))))
    if (!(yield* store.exists(digest))) return yield* new InfrastructureFailure({ operation: "probe-upload", message: "Uploaded blob was not observable" })
    yield* downloadObject(digest, join(stage, "recovered"))
    if (Number((yield* fs.stat(join(stage, "recovered"))).size) !== bytes) return yield* new InfrastructureFailure({ operation: "probe-download", message: "Recovered length differed" })
  }).pipe(Effect.provide(azureArtifactStore(config)))
  yield* fs.writeFileString(report, yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ passed: true, digest, bytes, account: config.account, container: config.container }))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, FetchHttpClient.layer])))
