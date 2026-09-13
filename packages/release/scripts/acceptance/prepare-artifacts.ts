import { FileSystem, FetchHttpClient } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { ReleaseArtifactSchema } from "../../src/contracts"
import { decodePublisherPrivateKey, PublisherKeyId, SignedUpdateManifest, UpdateManifest } from "../../src/hosted-update/manifest"
import { prepareHostedRelease } from "../../src/hosted-update/publication"

class AcceptancePreparationFailed extends Schema.TaggedError<AcceptancePreparationFailed>()("AcceptancePreparationFailed", {}) {}
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* Config.string("MAGNITUDE_ACCEPTANCE_ARTIFACTS")
  const version = yield* Config.literal("0.0.14", "0.0.15")("MAGNITUDE_ACCEPTANCE_VERSION")
  const commit = yield* Config.string("MAGNITUDE_ACCEPTANCE_COMMIT")
  const key = yield* decodePublisherPrivateKey(yield* Config.string("DISTRIBUTION_ACCEPTANCE_PUBLISHER_PRIVATE_KEY"))
  const artifacts = yield* Effect.forEach((yield* fs.readDirectory(directory)).filter(name => name.endsWith(".artifact.json")), name => Effect.gen(function* () {
    const artifact = yield* Schema.decodeUnknown(Schema.parseJson(ReleaseArtifactSchema))(yield* fs.readFileString(join(directory, name)))
    if (artifact.kind !== "desktop" || Option.isNone(artifact.host)) return yield* new AcceptancePreparationFailed()
    const host = artifact.host.value
    if (host.startsWith("darwin-") ? !/\.(dmg|zip)$/.test(artifact.filename)
      : !host.startsWith("linux-") || !/\.(deb|rpm)$/.test(artifact.filename)) return yield* new AcceptancePreparationFailed()
    const target = host.startsWith("darwin-") ? { os: "darwin", arch: host.endsWith("arm64") ? "arm64" : "x64", package: artifact.filename.endsWith(".dmg") ? "dmg" : "mac-zip" }
      : { os: "linux", arch: host.includes("arm64") ? "arm64" : "x64", package: artifact.filename.endsWith(".deb") ? "deb" : "rpm" }
    const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version, commit,
      artifact: { id: artifact.id, target, path: `releases/${version}/acceptance-${commit}-${artifact.filename}`, bytes: artifact.bytes, sha256: artifact.sha256 },
    })
    return { file: join(directory, artifact.filename), manifest }
  }))
  const envelopes = yield* prepareHostedRelease({ artifacts, keyId: PublisherKeyId.make("acceptance"), privateKey: key,
    storageOrigin: "https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com", token: yield* Config.string("DISTRIBUTION_BLOB_READ_WRITE_TOKEN"),
  })
  yield* fs.writeFileString(join(directory, "prepared-manifests.json"), yield* Schema.encode(Schema.parseJson(Schema.Array(SignedUpdateManifest)))(envelopes))
  yield* Effect.logInfo("Acceptance artifacts uploaded, fully downloaded and publisher-signed; no channel promoted", { version, artifacts: envelopes.length })
})
BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
