import { FileSystem, Path } from "@effect/platform"
import { Database } from "bun:sqlite"
import { Context, DateTime, Effect, Layer, Option, PubSub, Schema, Stream } from "effect"
import {
  ServingModelId,
  ServingUsageRequest,
  ServingUsageSnapshot,
  type MeasuredDrafterAcceptance,
} from "@magnitudedev/acn-protocol"
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
  drafter: Schema.NullOr(Schema.NonEmptyString),
  draftTokens: Schema.NullOr(Count),
  acceptedDraftTokens: Schema.NullOr(Count),
})
export type ServingUsageRecord = typeof ServingUsageRecord.Type

/** Measured acceptance and decode speed for one model and drafter pair. */
export interface DrafterAcceptanceMeasurement {
  readonly model: string
  readonly drafter: string
  readonly acceptance: MeasuredDrafterAcceptance
}
export class ServingUsageFailed extends Schema.TaggedError<ServingUsageFailed>()("ServingUsageFailed", { message: Schema.String }) {}
export interface ServingUsage {
  readonly record: (record: ServingUsageRecord) => Effect.Effect<void>
  readonly read: (request: ServingUsageRequest) => Effect.Effect<ServingUsageSnapshot>
  readonly acceptance: Effect.Effect<readonly DrafterAcceptanceMeasurement[]>
  readonly changes: Stream.Stream<void>
}
export const ServingUsage = Context.GenericTag<ServingUsage>("@magnitudedev/acn/ServingUsage")

const operation = <A>(run: () => A) => Effect.try({ try: run, catch: () => new ServingUsageFailed({ message: "Local usage history could not be read or saved." }) })

interface DrafterRow { readonly model: string; readonly drafter: string; readonly acceptanceRate: number | null; readonly tokensPerSecond: number | null }
interface BaselineRow { readonly model: string; readonly tokensPerSecond: number | null }

/** A pair needs drafted tokens and a decode-speed sample; plain decoding of the same model is the baseline. */
export const projectDrafterAcceptance = (
  drafted: readonly DrafterRow[],
  baseline: readonly BaselineRow[],
): readonly DrafterAcceptanceMeasurement[] => {
  const plain = new Map(baseline.flatMap(({ model, tokensPerSecond }) =>
    tokensPerSecond !== null && Number.isFinite(tokensPerSecond) ? [[model, tokensPerSecond] as const] : []))
  return drafted.flatMap(({ model, drafter, acceptanceRate, tokensPerSecond }) => {
    if (acceptanceRate === null || tokensPerSecond === null || !Number.isFinite(tokensPerSecond)) return []
    const without = Option.fromNullable(plain.get(model))
    return [{
      model,
      drafter,
      acceptance: {
        acceptanceRate: Math.min(1, Math.max(0, acceptanceRate)),
        tokensPerSecond,
        withoutDrafterTokensPerSecond: without,
        slower: Option.exists(without, (value) => tokensPerSecond < value),
      },
    }]
  })
}

/** Version 1 adds the drafter identity and its drafted and accepted tokens to every observation. */
const migrate = (db: Database) => {
  const { user_version: version } = db.query("PRAGMA user_version").get() as { readonly user_version: number }
  if (version >= 1) return
  db.transaction(() => {
    const columns = new Set((db.query("PRAGMA table_info(usage)").all() as Array<{ readonly name: string }>).map(({ name }) => name))
    if (!columns.has("drafter")) db.exec("ALTER TABLE usage ADD COLUMN drafter TEXT")
    if (!columns.has("draft_tokens")) db.exec("ALTER TABLE usage ADD COLUMN draft_tokens INTEGER")
    if (!columns.has("accepted_draft_tokens")) db.exec("ALTER TABLE usage ADD COLUMN accepted_draft_tokens INTEGER")
    db.exec("CREATE INDEX IF NOT EXISTS usage_model_drafter ON usage(model, drafter)")
    db.exec("PRAGMA user_version = 1")
  })()
}

/** Completed observations are durable independently of the renderer and model residency. */
export const ServingUsageLive = Layer.scoped(ServingUsage, Effect.gen(function* () {
  const storage = yield* GlobalStorage
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  let database: Database | undefined
  let recordingFailures = 0
  const lock = yield* Effect.makeSemaphore(1)
  const recorded = yield* PubSub.sliding<void>(1)
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
        migrate(db)
        db.query("INSERT OR IGNORE INTO metadata VALUES (1, ?)").run(Date.now())
        database = db
        return db
      } catch (error) { db.close(); throw error }
    })
  })
  const record = (value: ServingUsageRecord) => lock.withPermits(1)(Effect.gen(function* () {
    const db = yield* open
    const record = yield* Schema.decodeUnknown(ServingUsageRecord)(value).pipe(Effect.mapError(() => new ServingUsageFailed({ message: "Invalid usage evidence." })))
    yield* operation(() => db.query(`INSERT OR IGNORE INTO usage
      (id, model, completed_at, input, cached, output, generation_ms, first_token_ms, complete,
        drafter, draft_tokens, accepted_draft_tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      record.id, record.model, record.completedAt, record.input, record.cached, record.output,
      record.generationMs, record.firstTokenMs, Number(record.complete),
      record.drafter, record.draftTokens, record.acceptedDraftTokens,
    ))
    yield* PubSub.publish(recorded, undefined)
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
      // One indexed interval per local calendar day keeps DST and fractional-offset zones exact.
      // Activity spans 53 Sunday-first weeks through today, independently of the summary period.
      const today = DateTime.startOf(zone.value, "day")
      const start = DateTime.subtract(DateTime.startOf(today, "week", { weekStartsOn: 0 }), { weeks: 52 })
      const days: Array<readonly [string, number, number]> = []
      for (let day = start; DateTime.toEpochMillis(day) <= DateTime.toEpochMillis(today); day = DateTime.add(day, { days: 1 })) {
        days.push([DateTime.formatIsoDate(day), DateTime.toEpochMillis(day), DateTime.toEpochMillis(DateTime.add(day, { days: 1 }))])
      }
      const dailyActivity = db.query(`WITH days(date, start, end) AS (VALUES ${days.map(() => "(?, ?, ?)").join(",")})
        SELECT days.date, COALESCE(SUM(COALESCE(usage.input, 0) + COALESCE(usage.output, 0)), 0) AS totalTokens
        FROM days LEFT JOIN usage ON usage.completed_at >= days.start AND usage.completed_at < days.end
          AND (? IS NULL OR usage.model = ?)
        GROUP BY days.date ORDER BY days.date`).all(...days.flat(), model, model)
      const models = db.query("SELECT model AS id, COUNT(*) AS requests FROM usage GROUP BY model ORDER BY model").all()
      const metadata = db.query("SELECT since FROM metadata WHERE id = 1").get()
      return Schema.decodeUnknownSync(ServingUsageSnapshot)({ _tag: "Available", ...metadata!, ...totals!, dailyActivity, models, recordingFailures })
    })
  })).pipe(Effect.catchAll(error => Effect.succeed({ _tag: "Unavailable" as const, message: error.message })))
  const acceptance = lock.withPermits(1)(Effect.gen(function* () {
    const db = yield* open
    return yield* operation(() => projectDrafterAcceptance(
      db.query(`SELECT model, drafter,
        SUM(accepted_draft_tokens) * 1.0 / SUM(draft_tokens) AS acceptanceRate,
        SUM(CASE WHEN generation_ms > 0 THEN output END) * 1000.0 / SUM(CASE WHEN output IS NOT NULL THEN generation_ms END) AS tokensPerSecond
        FROM usage WHERE drafter IS NOT NULL AND draft_tokens > 0 AND accepted_draft_tokens IS NOT NULL
        GROUP BY model, drafter ORDER BY model, drafter`).all() as DrafterRow[],
      db.query(`SELECT model,
        SUM(CASE WHEN generation_ms > 0 THEN output END) * 1000.0 / SUM(CASE WHEN output IS NOT NULL THEN generation_ms END) AS tokensPerSecond
        FROM usage WHERE drafter IS NULL GROUP BY model`).all() as BaselineRow[],
    ))
  })).pipe(Effect.catchAll((error) => Effect.logWarning(error.message).pipe(Effect.as([]))))
  return ServingUsage.of({
    record,
    read,
    acceptance,
    changes: Stream.fromPubSub(recorded),
  })
}))
