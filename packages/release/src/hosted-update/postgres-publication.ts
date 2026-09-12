import { Effect, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import type { Pool } from "pg"
import { isNewerVersion, releaseChannelOf } from "../client-update/release-channels"
import { SignedUpdateManifest, verifyUpdateManifest } from "./manifest"
import { DistributionNamespace } from "./postgres-store"
import { ReleasePublicationBatch, ReleasePublicationFailed, ReleasePublicationStore } from "./publication"

/** The publisher has separate credentials; request runtimes cannot mutate releases or channels. */
export const postgresReleasePublicationStore = (pool: Pool, namespace: typeof DistributionNamespace.Type,
  trustedPublishers: ReadonlyMap<string, KeyObject>): ReleasePublicationStore => ({
  promote: envelopes => Effect.gen(function* () {
    const manifests = yield* Effect.forEach(envelopes, envelope => verifyUpdateManifest(envelope, trustedPublishers)).pipe(
      Effect.mapError(() => new ReleasePublicationFailed({ stage: "batch" })))
    yield* Schema.decodeUnknown(ReleasePublicationBatch)(manifests.map(manifest => ({ file: "", manifest }))).pipe(
      Effect.mapError(() => new ReleasePublicationFailed({ stage: "batch" })))
    const first = manifests[0]!, channel = releaseChannelOf(first.version)
    if (channel === "unknown") return yield* new ReleasePublicationFailed({ stage: "batch" })
    yield* Effect.acquireUseRelease(
      Effect.tryPromise({ try: () => pool.connect(), catch: () => new ReleasePublicationFailed({ stage: "database" }) }).pipe(
        Effect.map(client => ({ client, discard: false }))),
      connection => {
        const { client } = connection
        const query = (sql: string, values: readonly unknown[] = []) => Effect.tryPromise({
          try: () => client.query(sql.replaceAll("magnitude_distribution.", `${namespace}.`), [...values]),
          catch: () => new ReleasePublicationFailed({ stage: "database" }),
        })
        return Effect.gen(function* () {
          yield* query("BEGIN")
          yield* query("SET LOCAL statement_timeout = '10s'")
          yield* query("SELECT pg_advisory_xact_lock(hashtext($1))", [namespace])
          yield* query(`INSERT INTO magnitude_distribution.releases(version,channel,source_commit) VALUES ($1,$2,$3) ON CONFLICT DO NOTHING`, [first.version, channel, first.commit])
          const release = yield* query("SELECT source_commit,channel,withdrawn FROM magnitude_distribution.releases WHERE version=$1", [first.version])
          const releaseRows = yield* Schema.decodeUnknown(Schema.Array(Schema.Struct({ source_commit: Schema.String, channel: Schema.String, withdrawn: Schema.Boolean })))(release.rows).pipe(
            Effect.mapError(() => new ReleasePublicationFailed({ stage: "conflict" })))
          if (releaseRows.length !== 1 || releaseRows[0]!.source_commit !== first.commit || releaseRows[0]!.channel !== channel || releaseRows[0]!.withdrawn) {
            return yield* new ReleasePublicationFailed({ stage: "conflict" })
          }
          for (let index = 0; index < manifests.length; index++) {
            const manifest = manifests[index]!, envelope = envelopes[index]!, artifact = manifest.artifact, target = artifact.target
            yield* query(`INSERT INTO magnitude_distribution.artifacts(release_version,artifact_id,os,arch,package,envelope)
              VALUES ($1,$2,$3,$4,$5,$6::jsonb) ON CONFLICT DO NOTHING`, [manifest.version, artifact.id, target.os, target.arch, target.package, JSON.stringify(envelope)])
            const existing = yield* query("SELECT envelope FROM magnitude_distribution.artifacts WHERE release_version=$1 AND artifact_id=$2", [manifest.version, artifact.id])
            const stored = yield* Schema.decodeUnknown(Schema.Struct({ envelope: SignedUpdateManifest }))(existing.rows[0]).pipe(
              Effect.mapError(() => new ReleasePublicationFailed({ stage: "conflict" })))
            if (stored.envelope.payload !== envelope.payload || stored.envelope.signature !== envelope.signature || stored.envelope.keyId !== envelope.keyId) {
              return yield* new ReleasePublicationFailed({ stage: "conflict" })
            }
            const previous = yield* query("SELECT release_version FROM magnitude_distribution.channels WHERE channel=$1 AND os=$2 AND arch=$3 AND package=$4", [channel, target.os, target.arch, target.package])
            const previousRows = yield* Schema.decodeUnknown(Schema.Array(Schema.Struct({ release_version: Schema.String })))(previous.rows).pipe(
              Effect.mapError(() => new ReleasePublicationFailed({ stage: "conflict" })))
            const previousVersion = previousRows[0]?.release_version
            if (previousVersion !== undefined && previousVersion !== manifest.version && !isNewerVersion(manifest.version, previousVersion)) {
              return yield* new ReleasePublicationFailed({ stage: "conflict" })
            }
            yield* query(`INSERT INTO magnitude_distribution.channels(channel,os,arch,package,release_version,artifact_id)
              VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (channel,os,arch,package) DO UPDATE
              SET release_version=EXCLUDED.release_version,artifact_id=EXCLUDED.artifact_id`, [channel, target.os, target.arch, target.package, manifest.version, artifact.id])
          }
          yield* query("COMMIT")
        }).pipe(Effect.onError(() => query("ROLLBACK").pipe(Effect.catchAll(() => Effect.sync(() => { connection.discard = true })))), Effect.uninterruptible)
      },
      connection => Effect.sync(() => connection.client.release(connection.discard)),
    )
  }),
})
