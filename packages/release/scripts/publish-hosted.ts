import { FileSystem, FetchHttpClient } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Redacted, Runtime } from "effect"
import { createPublicKey } from "node:crypto"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { Pool } from "pg"
import { acquireRelease, releaseUrl } from "../src/acquisition"
import { decodePublisherPrivateKey, decodePublisherPublicKey, PublisherKeyId } from "../src/hosted-update/manifest"
import { hostedDesktopManifests, HostedCandidateInvalid } from "../src/hosted-update/release-candidate"
import { downloadUpdateArtifact } from "../src/hosted-update/installer-download"
import { postgresReleasePublicationStore } from "../src/hosted-update/postgres-publication"
import { publishHostedRelease, ReleasePublicationFailed, ReleasePublicationStore } from "../src/hosted-update/publication"

const run = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const version = yield* Config.string("MAGNITUDE_RELEASE_VERSION")
  const commit = yield* Config.string("MAGNITUDE_SOURCE_COMMIT")
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hosted-release-" })
  // The native publication gate has already accepted and published these exact bytes. Recovery
  // uses the same public release, never a rebuild or an arbitrary local installer directory.
  const baseUrl = "https://github.com/magnitudedev/magnitude/releases/download"
  const { manifest: release } = yield* acquireRelease(baseUrl, version, join(directory, "manifest"))
  const manifests = yield* hostedDesktopManifests(release, commit)
  const privateKey = yield* decodePublisherPrivateKey(Redacted.value(yield* Config.redacted("DISTRIBUTION_PUBLISHER_PRIVATE_KEY")))
  const publicKey = yield* decodePublisherPublicKey(yield* fs.readFileString(fileURLToPath(new URL("../resources/distribution/magnitude-2026-01.pub.pem", import.meta.url))))
  if (!createPublicKey(privateKey).equals(publicKey)) return yield* new HostedCandidateInvalid({ message: "Publisher credential differs from application-embedded trust" })
  const artifacts = yield* Effect.forEach(manifests, manifest => Effect.gen(function* () {
    const filename = manifest.artifact.path.split("/").at(-1)!
    const file = join(directory, filename)
    yield* downloadUpdateArtifact({ manifest, destination: file, url: releaseUrl(baseUrl, version, filename), onProgress: Option.none() })
    return { file, manifest }
  }), { concurrency: 2 })
  const databaseUrl = Redacted.value(yield* Config.redacted("DISTRIBUTION_PUBLISHER_DATABASE_URL"))
  const ca = yield* Config.string("DISTRIBUTION_DATABASE_CA")
  const runtime = yield* Effect.runtime<never>()
  const pool = yield* Effect.acquireRelease(Effect.try({ try: () => {
    const url = new URL(databaseUrl)
    if (!["postgres:", "postgresql:"].includes(url.protocol) || !ca.includes("BEGIN CERTIFICATE")) throw new Error("Invalid publisher configuration")
    const pool = new Pool({ host: url.hostname, port: Number(url.port || 5432), user: decodeURIComponent(url.username), password: decodeURIComponent(url.password),
      database: url.pathname.slice(1), ssl: { ca, rejectUnauthorized: true }, max: 2, connectionTimeoutMillis: 5000, idleTimeoutMillis: 30000 })
    pool.on("error", () => { Runtime.runSync(runtime)(Effect.logWarning("Publisher database connection closed")) })
    return pool
  }, catch: () => new ReleasePublicationFailed({ stage: "database" }) }), pool => Effect.promise(() => pool.end()))
  const envelopes = yield* publishHostedRelease({ artifacts, keyId: PublisherKeyId.make("magnitude-2026-01"), privateKey,
    storageOrigin: "https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com",
    token: Redacted.value(yield* Config.redacted("DISTRIBUTION_BLOB_READ_WRITE_TOKEN")),
  }).pipe(Effect.provideService(ReleasePublicationStore, postgresReleasePublicationStore(pool, "magnitude_distribution", new Map([["magnitude-2026-01", publicKey]]))))
  yield* Effect.logInfo("Accepted desktop release published to Magnitude", { version, artifacts: envelopes.length })
}))
BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
