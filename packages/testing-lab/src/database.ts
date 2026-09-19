import { Context, Effect, Exit, Layer, Redacted, Schema } from "effect"
import { Pool, type PoolClient, type QueryResultRow } from "pg"
import { InfrastructureFailure } from "./domain"

export interface DatabaseSession {
  readonly query: (sql: string, values?: readonly unknown[]) => Effect.Effect<ReadonlyArray<QueryResultRow>, InfrastructureFailure>
}
export interface Database extends DatabaseSession {
  readonly transaction: <A, E, R>(body: (session: DatabaseSession) => Effect.Effect<A, E, R>) => Effect.Effect<A, E | InfrastructureFailure, R>
}
export const Database = Context.GenericTag<Database>("@magnitudedev/testing-lab/Database")

// pg is the sole asynchronous database boundary. Error text deliberately excludes SQL/credentials.
const failed = () => new InfrastructureFailure({ operation: "database", message: "PostgreSQL operation failed" })
const session = (client: Pool | PoolClient): DatabaseSession => ({
  query: (sql, values = []) => Effect.tryPromise({ try: () => client.query(sql, [...values]), catch: failed }).pipe(Effect.map(result => result.rows)),
})
export const databaseLayer = (connectionString: Redacted.Redacted<string>) => Layer.scoped(Database, Effect.gen(function* () {
  const pool = yield* Effect.acquireRelease(Effect.sync(() => new Pool({ connectionString: Redacted.value(connectionString), max: 8,
    connectionTimeoutMillis: 10_000, idleTimeoutMillis: 30_000, statement_timeout: 30_000 })),
  pool => Effect.promise(() => pool.end()))
  return {
    ...session(pool),
    transaction: <A, E, R>(body: (db: DatabaseSession) => Effect.Effect<A, E, R>) => Effect.scoped(Effect.gen(function* () {
      const client = yield* Effect.acquireRelease(Effect.tryPromise({ try: () => pool.connect(), catch: failed }),
        client => Effect.sync(() => client.release()))
      const db = session(client)
      return yield* Effect.acquireUseRelease(db.query("BEGIN"), () => body(db), (_, exit) =>
        db.query(Exit.isSuccess(exit) ? "COMMIT" : "ROLLBACK").pipe(Effect.orDie))
    })),
  } satisfies Database
}))

export const initializeDatabase = Effect.flatMap(Database, db => db.transaction(tx => Effect.gen(function* () {
  yield* tx.query("SELECT pg_advisory_xact_lock(91826001)")
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_objects (
    owner text NOT NULL, digest text NOT NULL, bytes bigint NOT NULL CHECK(bytes >= 0), PRIMARY KEY(owner,digest)
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_inputs (
    owner text NOT NULL, digest text NOT NULL, kind text NOT NULL CHECK(kind IN ('source','artifacts')), PRIMARY KEY(owner,digest),
    FOREIGN KEY(owner,digest) REFERENCES lab_objects(owner,digest)
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_leases (
    lease_id text PRIMARY KEY, run_id text NOT NULL, target_id text NOT NULL,
    provider text NOT NULL, resource_name text NOT NULL, state text NOT NULL,
    expires_at timestamptz NOT NULL, fence bigint NOT NULL DEFAULT 1,
    worker text NOT NULL, claim_expires_at timestamptz NOT NULL,
    UNIQUE(provider, resource_name), CHECK (fence > 0),
    CHECK (state IN ('Allocating','Ready','Releasing','Released'))
  )`)
  yield* tx.query(`CREATE UNIQUE INDEX IF NOT EXISTS lab_one_spark_lease ON lab_leases(provider)
    WHERE provider = 'spark' AND state <> 'Released'`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_runs (
    run_id text PRIMARY KEY, owner text NOT NULL, idempotency_key text NOT NULL, request_digest text NOT NULL,
    plan text NOT NULL, result text, state text NOT NULL CHECK(state IN ('Queued','Running','Cancelling','Finished')),
    reserved_usd double precision NOT NULL CHECK(reserved_usd > 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(), deadline timestamptz NOT NULL,
    UNIQUE(owner,idempotency_key)
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_work (
    run_id text NOT NULL REFERENCES lab_runs(run_id), target_id text NOT NULL,
    state text NOT NULL CHECK(state IN ('Queued','Running','Finished')),
    fence bigint NOT NULL DEFAULT 1, worker text, claim_expires_at timestamptz,
    attempts integer NOT NULL DEFAULT 0, result text,
    PRIMARY KEY(run_id,target_id)
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_attempts (
    run_id text NOT NULL, target_id text NOT NULL, fence bigint NOT NULL, worker text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(), ended_at timestamptz, detail text,
    PRIMARY KEY(run_id,target_id,fence), FOREIGN KEY(run_id,target_id) REFERENCES lab_work(run_id,target_id)
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_events (
    cursor bigserial PRIMARY KEY, run_id text NOT NULL REFERENCES lab_runs(run_id),
    kind text NOT NULL, detail text NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp()
  )`)
  yield* tx.query(`CREATE TABLE IF NOT EXISTS lab_worker_tickets (
    ticket_id uuid PRIMARY KEY, token_digest text NOT NULL UNIQUE,
    run_id text NOT NULL, target_id text NOT NULL, fence bigint NOT NULL, worker text NOT NULL,
    invocation text NOT NULL, revoked boolean NOT NULL DEFAULT false,
    UNIQUE(run_id,target_id,fence), FOREIGN KEY(run_id,target_id) REFERENCES lab_work(run_id,target_id)
  )`)
})))

/** Decode every database boundary rather than asserting driver output types. */
export const decodeRow = <A, I>(schema: Schema.Schema<A, I>, row: unknown) => Schema.decodeUnknown(schema)(row).pipe(
  Effect.mapError(() => new InfrastructureFailure({ operation: "database-decode", message: "Stored lab record does not match its schema" })),
)
