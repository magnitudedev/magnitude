import { FileSystem, Path } from "@effect/platform"
import { Database } from "bun:sqlite"
import { Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import { ServingModelId, ServingUsageRequest, ServingUsageSnapshot } from "@magnitudedev/acn-protocol"
import { GlobalStorage } from "@magnitudedev/storage"

const Count = Schema.Number.pipe(Schema.int(), Schema.nonNegative())
export const ServingUsageRecord = Schema.Struct({
  id: Schema.NonEmptyString.pipe(Schema.brand("ServingUsageRecordId")),
  model: ServingModelId,
  completedAt: Schema.Number,
  input: Schema.NullOr(Count),
  cached: Schema.NullOr(Count),
  output: Schema.NullOr(Count),
  generationMs: Schema.NullOr(Schema.Number.pipe(Schema.positive())),
  firstTokenMs: Schema.NullOr(Schema.Number.pipe(Schema.nonNegative())),
  complete: Schema.Boolean,
})
export type ServingUsageRecord = typeof ServingUsageRecord.Type
export class ServingUsageFailed extends Schema.TaggedError<ServingUsageFailed>()("ServingUsageFailed", { message: Schema.String }) {}
export interface ServingUsage {
  readonly record: (record: ServingUsageRecord) => Effect.Effect<void>
  readonly read: (request: ServingUsageRequest) => Effect.Effect<ServingUsageSnapshot>
}
export const ServingUsage = Context.GenericTag<ServingUsage>("@magnitudedev/acn/ServingUsage")

const operation = <A>(run: () => A) => Effect.try({ try: run, catch: () => new ServingUsageFailed({ message: "Local usage history could not be read or saved." }) })

/** Completed observations are durable independently of the renderer and model residency. */
export const ServingUsageLive = Layer.scoped(ServingUsage, Effect.gen(function* () {
  const storage = yield* GlobalStorage
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  let database: Database | undefined
  let recordingFailures = 0
  const lock = yield* Effect.makeSemaphore(1)
  yield* Effect.addFinalizer(() => Effect.sync(() => database?.close()))
  const open = Effect.gen(function* () {
    if (database) return database
    yield* fs.makeDirectory(storage.root, { recursive: true }).pipe(Effect.mapError(() => new ServingUsageFailed({ message: "Usage storage is unavailable." })))
    return yield* operation(() => {
      const db = new Database(path.join(storage.root, "serving-usage.sqlite"), { create: true, strict: true })
      try {
        db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS usage (
            id TEXT PRIMARY KEY, model TEXT NOT NULL, completed_at REAL NOT NULL,
            input INTEGER, cached INTEGER, output INTEGER,
            generation_ms REAL, first_token_ms REAL, complete INTEGER NOT NULL
          );
          CREATE INDEX IF NOT EXISTS usage_date ON usage(completed_at);
          CREATE INDEX IF NOT EXISTS usage_model_date ON usage(model, completed_at);
          CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id = 1), since REAL NOT NULL);`)
        db.query("INSERT OR IGNORE INTO metadata VALUES (1, ?)").run(Date.now())
        database = db
        return db
      } catch (error) { db.close(); throw error }
    })
  })
  const record = (value: ServingUsageRecord) => lock.withPermits(1)(Effect.gen(function* () {
    const db = yield* open
    const record = yield* Schema.decodeUnknown(ServingUsageRecord)(value).pipe(Effect.mapError(() => new ServingUsageFailed({ message: "Invalid usage evidence." })))
    yield* operation(() => db.query("INSERT OR IGNORE INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)").run(
      record.id, record.model, record.completedAt, record.input, record.cached, record.output,
      record.generationMs, record.firstTokenMs, Number(record.complete),
    ))
  })).pipe(Effect.catchAll(error => Effect.sync(() => { recordingFailures++ }).pipe(Effect.zipRight(Effect.logError(error.message)))))
  const read = (request: ServingUsageRequest) => lock.withPermits(1)(Effect.gen(function* () {
    const db = yield* open
    return yield* operation(() => {
      const zone = DateTime.setZoneNamed(DateTime.unsafeNow(), request.timeZone)
      if (Option.isNone(zone)) throw new Error("Invalid time zone")
      const since = request.period === "Today" ? DateTime.toEpochMillis(DateTime.startOf(zone.value, "day")) : 0
      const model = Option.getOrNull(request.model)
      const totals = db.query(`SELECT COUNT(*) AS requests,
        COALESCE(SUM(complete = 0), 0) AS incompleteRequests,
        COALESCE(SUM(input), 0) AS inputTokens, COALESCE(SUM(cached), 0) AS cachedInputTokens,
        COALESCE(SUM(output), 0) AS outputTokens,
        COALESCE(SUM(COALESCE(input, 0) + COALESCE(output, 0)), 0) AS totalTokens,
        COUNT(cached) AS cachedInputRequests,
        SUM(CASE WHEN generation_ms > 0 THEN output END) * 1000.0 / SUM(generation_ms) AS tokensPerSecond,
        AVG(first_token_ms) AS timeToFirstTokenMs, COUNT(generation_ms) AS speedSamples,
        COUNT(first_token_ms) AS latencySamples
        FROM usage WHERE completed_at >= ? AND (? IS NULL OR model = ?)`)
        .get(since, model, model)
      const models = db.query("SELECT model AS id, COUNT(*) AS requests FROM usage GROUP BY model ORDER BY model").all()
      const metadata = db.query("SELECT since FROM metadata WHERE id = 1").get()
      return Schema.decodeUnknownSync(ServingUsageSnapshot)({ _tag: "Available", ...metadata!, ...totals!, models, recordingFailures })
    })
  })).pipe(Effect.catchAll(error => Effect.succeed({ _tag: "Unavailable" as const, message: error.message })))
  return ServingUsage.of({ record, read })
}))
