import { Effect, Option, Schema } from "effect"
import type { Pool } from "pg"
import { DistributionStore, DistributionStoreUnavailable, type CheckObservation } from "./service"
import { SignedUpdateManifest } from "./manifest"

export const DistributionNamespace = Schema.Literal("magnitude_distribution", "magnitude_distribution_acceptance")

const installationUpsert = `
  INSERT INTO magnitude_distribution.installations (installation_id, version, os, os_version, arch, package, distro, distro_version, country)
  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
  ON CONFLICT (installation_id) DO UPDATE SET last_seen = now(), version = EXCLUDED.version,
    os = EXCLUDED.os, os_version = EXCLUDED.os_version, arch = EXCLUDED.arch, package = EXCLUDED.package,
    distro = EXCLUDED.distro, distro_version = EXCLUDED.distro_version, country = EXCLUDED.country
  RETURNING installation_id
`
const installationParameters = ({ installation, request, country }: Pick<typeof CheckObservation.Type, "installation" | "request" | "country">) =>
  [installation, request.version, request.os, request.os_version, request.arch, request.package,
    Option.getOrElse(request.distro, () => ""), Option.getOrElse(request.distro_version, () => ""), Option.getOrElse(country, () => "")]

/** pg is the single intentional Promise boundary. Queries never include credentials or raw request data in errors. */
export const postgresDistributionStore = (pool: Pool, namespace: typeof DistributionNamespace.Type = "magnitude_distribution"): DistributionStore => {
  const query = (text: string, values: readonly unknown[]) => Effect.tryPromise({
    try: () => pool.query(text.replaceAll("magnitude_distribution.", `${namespace}.`), [...values]),
    catch: () => new DistributionStoreUnavailable(),
  })
  return DistributionStore.of({
    admit: (installation, nonce, expiresAt) => query(`
      INSERT INTO magnitude_distribution.request_nonces (installation_id, nonce, expires_at)
      VALUES ($1, $2, to_timestamp($3)) ON CONFLICT DO NOTHING RETURNING installation_id
    `, [installation, nonce, expiresAt]).pipe(Effect.map(result => result.rowCount === 1)),
    candidates: request => query(`
      SELECT a.envelope FROM magnitude_distribution.channels c
      JOIN magnitude_distribution.artifacts a ON a.release_version = c.release_version AND a.artifact_id = c.artifact_id
      JOIN magnitude_distribution.releases r ON r.version = c.release_version
      WHERE NOT r.withdrawn AND c.os = $1 AND c.arch = $2 AND c.package = $3
    `, [request.os, request.arch, request.package]).pipe(Effect.flatMap(result =>
      Schema.decodeUnknown(Schema.Array(Schema.Struct({ envelope: SignedUpdateManifest })))(result.rows).pipe(
        Effect.map(rows => rows.map(row => row.envelope)), Effect.mapError(() => new DistributionStoreUnavailable()),
      ))),
    recordCheck: observation => {
      const parameters = [...installationParameters(observation), Option.getOrNull(observation.offeredVersion)]
      return query(`
        WITH installation AS (${installationUpsert})
        INSERT INTO magnitude_distribution.installation_daily (installation_id, version, os, os_version, arch, package, distro, distro_version, country, offered_version)
        SELECT installation_id,$2,$3,$4,$5,$6,$7,$8,$9,$10 FROM installation
        ON CONFLICT (day, installation_id, version, os, os_version, arch, package, distro, distro_version, country)
        DO UPDATE SET checks = magnitude_distribution.installation_daily.checks + 1,
          last_seen = now(), offered_version = EXCLUDED.offered_version
      `, parameters).pipe(Effect.asVoid)
    },
    artifact: (release, id) => query(`
      SELECT a.envelope FROM magnitude_distribution.artifacts a
      JOIN magnitude_distribution.releases r ON r.version = a.release_version
      WHERE a.release_version = $1 AND a.artifact_id = $2 AND NOT r.withdrawn
    `, [release, id]).pipe(Effect.flatMap(result => Schema.decodeUnknown(Schema.Array(Schema.Struct({ envelope: SignedUpdateManifest })))(result.rows).pipe(
      Effect.map(rows => Option.map(Option.fromNullable(rows[0]), row => row.envelope)), Effect.mapError(() => new DistributionStoreUnavailable()),
    ))),
    recordDownload: observation => query(`
      WITH installation AS (${installationUpsert})
      INSERT INTO magnitude_distribution.artifact_daily (installation_id, version, os, os_version, arch, package, distro, distro_version, country, release_version, artifact_id)
      SELECT installation_id,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11 FROM installation
      ON CONFLICT (day, installation_id, version, os, os_version, arch, package, distro, distro_version, country, release_version, artifact_id)
      DO UPDATE SET requests = magnitude_distribution.artifact_daily.requests + 1, last_seen = now()
    `, [...installationParameters(observation), observation.release, observation.artifact]).pipe(Effect.asVoid),
  })
}
