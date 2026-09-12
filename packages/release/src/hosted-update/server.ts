import { Pool } from "pg"
import { Effect, Option, Runtime, Schema } from "effect"
import { decodePublisherPublicKey } from "./manifest"
import { handleUpdateCheck, DistributionStore } from "./service"
import { postgresDistributionStore } from "./postgres-store"
import { handleArtifactDownload } from "./download"

export const DistributionServerConfig = Schema.Struct({
  origin: Schema.String,
  databaseUrl: Schema.String,
  databaseCa: Schema.String,
  storageOrigin: Schema.String,
  publishers: Schema.Record({ key: Schema.String, value: Schema.String }),
})
export type DistributionServerConfig = typeof DistributionServerConfig.Type
export class DistributionConfigurationFailed extends Schema.TaggedError<DistributionConfigurationFailed>()("DistributionConfigurationFailed", {}) {}

/** Long-lived hosting composition; one small pool per serverless instance, never per request. */
export const makeDistributionServer = (config: DistributionServerConfig) => Effect.gen(function* () {
  const keys = yield* Effect.forEach(Object.entries(config.publishers), ([id, pem]) => decodePublisherPublicKey(pem).pipe(Effect.map(key => [id, key] as const)))
  const pool = yield* Effect.try({ try: () => {
    const url = new URL(config.databaseUrl)
    const origin = new URL(config.origin)
    const storage = new URL(config.storageOrigin)
    if (!["postgres:", "postgresql:"].includes(url.protocol) || origin.protocol !== "https:" || origin.origin !== config.origin
      || storage.protocol !== "https:" || storage.origin !== config.storageOrigin || !config.databaseCa.includes("BEGIN CERTIFICATE")) throw new Error("Invalid server configuration")
    return new Pool({
      host: url.hostname, port: Number(url.port || 5432), user: decodeURIComponent(url.username), password: decodeURIComponent(url.password), database: url.pathname.slice(1),
      ssl: { ca: config.databaseCa, rejectUnauthorized: true }, max: 5, connectionTimeoutMillis: 5000, idleTimeoutMillis: 30000, statement_timeout: 5000,
    })
  }, catch: () => new DistributionConfigurationFailed() })
  // Idle pool errors have no pending query to receive them. Never log connection strings or SQL details.
  const runtime = yield* Effect.runtime<never>()
  pool.on("error", () => { Runtime.runSync(runtime)(Effect.logWarning("Distribution database connection closed")) })
  const store = postgresDistributionStore(pool)
  return {
    pool,
    check: (request: Request, country: string | undefined) => handleUpdateCheck(request, {
      origin: config.origin, country: Option.fromNullable(country), trustedPublishers: new Map(keys),
    }).pipe(Effect.provideService(DistributionStore, store)),
    download: (request: Request, country: string | undefined) => handleArtifactDownload(request, {
      origin: config.origin, storageOrigin: config.storageOrigin, country: Option.fromNullable(country), trustedPublishers: new Map(keys),
    }).pipe(Effect.provideService(DistributionStore, store)),
  }
})

/** Hosting boundary for frameworks outside the Effect application. */
export const createDistributionServer = async (input: unknown) => {
  const server = await Effect.runPromise(Schema.decodeUnknown(DistributionServerConfig)(input).pipe(Effect.flatMap(makeDistributionServer)))
  return {
    pool: server.pool,
    check: (request: Request, country?: string) => Effect.runPromise(server.check(request, country)),
    download: (request: Request, country?: string) => Effect.runPromise(server.download(request, country)),
  }
}
