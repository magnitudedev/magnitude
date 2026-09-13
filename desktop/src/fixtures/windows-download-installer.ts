import { FetchHttpClient, FileSystem } from "@effect/platform"
import { NodeContext, NodeRuntime } from "@effect/platform-node"
import { Config, Effect, Option, Schema } from "effect"
import { createPublicKey } from "node:crypto"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { downloadUpdateArtifact, SignedUpdateManifest, verifyUpdateManifest } from "@magnitudedev/release/hosted-update"

NodeRuntime.runMain(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("MAGNITUDE_WINDOWS_ACCEPTANCE_ROOT")
  const version = yield* Config.string("MAGNITUDE_WINDOWS_ACCEPTANCE_FROM")
  const envelopes = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Tuple(SignedUpdateManifest)))(
    yield* fs.readFileString(join(root, version, "artifacts/prepared-manifests.json")))
  const trusted = new Map([["acceptance", createPublicKey(yield* fs.readFileString(fileURLToPath(new URL("../../../packages/release/resources/distribution/acceptance.pub.pem", import.meta.url))))]])
  const manifest = yield* verifyUpdateManifest(envelopes[0], trusted)
  if (manifest.version !== version || manifest.artifact.target.package !== "windows-exe") return yield* Effect.die("Wrong installer target")
  const result = yield* downloadUpdateArtifact({ manifest,
    url: new URL(manifest.artifact.path, "https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com/").href,
    destination: join(root, "consumer/downloaded-installer.exe"), onProgress: Option.none(),
  })
  yield* Effect.logInfo("Verified hosted installer transfer", { version, bytes: result.bytes, strategy: result.strategy })
}).pipe(Effect.provide([NodeContext.layer, FetchHttpClient.layer])))
