import { FileSystem, FetchHttpClient } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Redacted, Runtime, Schema } from "effect"
import { createPublicKey } from "node:crypto"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { Pool } from "pg"
import { acquireRelease } from "../src/acquisition"
import { decodePublisherPrivateKey, decodePublisherPublicKey, PublisherKeyId, PublishedUpdate } from "../src/hosted-update/manifest"
import { hostedDesktopManifests, HostedCandidateInvalid } from "../src/hosted-update/release-candidate"
import { verifyGithubRelease } from "../src/hosted-update/github-release"
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
  yield* verifyGithubRelease(manifests, Option.map(yield* Config.option(Config.redacted("GH_TOKEN")), Redacted.value))
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
  const envelopes = yield* publishHostedRelease({ artifacts: manifests, keyId: PublisherKeyId.make("magnitude-2026-01"), privateKey,
  }).pipe(Effect.provideService(ReleasePublicationStore, postgresReleasePublicationStore(pool, "magnitude_distribution", new Map([["magnitude-2026-01", publicKey]]))))
  const installationRecords = yield* Config.option(Config.string("MAGNITUDE_INSTALL_PUBLICATIONS_OUTPUT"))
  if (Option.isSome(installationRecords)) yield* fs.writeFileString(installationRecords.value,
    yield* Schema.encode(Schema.parseJson(Schema.Array(PublishedUpdate)))(envelopes))
  yield* Effect.logInfo("Accepted desktop release published to Magnitude", { version, artifacts: envelopes.length })
}))
BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
