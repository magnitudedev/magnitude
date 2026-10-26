import { FileSystem, Path } from "@effect/platform"
import type { PlatformError } from "@effect/platform/Error"
import { Array, Cause, Chunk, DateTime, Duration, Effect, Layer, Option, type ParseResult, Schema, ScopedCache, Stream } from "effect"
import type { SessionEvent, SessionId } from "@magnitudedev/acn-protocol"
import { GlobalStorage } from "../services/global-storage"
import { enableWal, type SqliteCheckpointIncomplete, type SqliteWalRefused, transaction, truncateWal, userVersion } from "../sqlite/database"
import { type SqliteBinding, type SqliteConnection, SqliteDriver, type SqliteDriverError } from "../sqlite/driver"
import {
  ChatStore,
  ChatStoreCorrupt,
  type ChatStoreError,
  ChatStoreIncompatible,
  ChatStoreRejected,
  ChatStoreUnavailable,
  type ChatSessionRecord,
  chatStoreVersion,
} from "./store"
import { StoredEvent } from "./stored-event"
import { cacheChange, createdRecord, renameOutcome } from "./session-record"
import { StoredSession } from "./stored-session"

// The file's format has one owner, its `user_version`, which is `chatStoreVersion` for this format.
// Any addition to the stored shape (a column, a JSON field, an event type, an event version) bumps
// `chatStoreVersion` and adds the upgrade from the previous format to `upgrades`. A file of this
// format or an earlier one opens, upgraded to this one; a file of a later format is refused as
// newer, since this build cannot know what that format added and would misread it.

const tables = ["events", "sessions"]

// `events` is a rowid table: output rows are kilobytes, and its `position` rowid is commit order;
// AUTOINCREMENT never reuses one. `version` is the event type's schema version and `created_at` the
// event's time in epoch ms; `data` is the event's other fields as JSON, never `type`, `version` or
// `time`, which the columns own. `events_session` serves a session's read in position order and its
// deletion. `sessions` holds each session's record (`stored-session.ts`), created in the transaction
// of the commit that holds its first message; `message_count` and `updated_at` cache the session's
// log, and are updated in the transaction of each commit to it. `sessions_updated_at` serves the list
// order.
const schema = [
  `CREATE TABLE events (
    position   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    type       TEXT    NOT NULL,
    version    INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    data       TEXT    NOT NULL
  )`,
  "CREATE INDEX events_session ON events (session_id, position)",
  `CREATE TABLE sessions (
    session_id    TEXT    PRIMARY KEY,
    title         TEXT    NOT NULL,
    message_count INTEGER NOT NULL,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
  )`,
  "CREATE INDEX sessions_updated_at ON sessions (updated_at DESC, session_id)",
]

/** How a file of one format becomes the next: that format's tables, which recognise it, and the statements that upgrade it. */
interface Upgrade {
  readonly tables: ReadonlyArray<string>
  readonly statements: ReadonlyArray<string>
}
// One upgrade for each earlier format, oldest first: a version bump does not compile until it adds its upgrade.
type Upgrades<Version extends number, Earlier extends ReadonlyArray<Upgrade> = []> =
  [...Earlier, Upgrade]["length"] extends Version ? Earlier : Upgrades<Version, [...Earlier, Upgrade]>
const upgrades: Upgrades<typeof chatStoreVersion> = []
const earlier: ReadonlyArray<Upgrade> = upgrades
// Each format's tables, by version from 1.
const formats = [...earlier.map((upgrade) => upgrade.tables), tables]
const stamp = `PRAGMA user_version=${chatStoreVersion}`

// Durable writes, and deleted content overwritten rather than left in free pages.
const pragmas = ["PRAGMA synchronous=FULL", "PRAGMA secure_delete=ON"]

const append = "INSERT INTO events (session_id, type, version, created_at, data) VALUES (?, ?, ?, ?, ?)"
// A commit holding a message creates the record of a session that has none.
const create = "INSERT INTO sessions (session_id, title, message_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (session_id) DO NOTHING"
// The latest observation never moves back: a rename touches it with the wall clock, which the next
// event may precede.
const cached = "UPDATE sessions SET message_count = message_count + ?, updated_at = max(updated_at, ?) WHERE session_id = ?"
const recorded = "SELECT session_id, title, message_count, created_at, updated_at FROM sessions"
const listed = `${recorded} ORDER BY updated_at DESC, session_id LIMIT ? OFFSET ?`

// Events per read page; a read holds one page in memory.
const page = 1000

const Tables = Schema.Array(Schema.Struct({ name: Schema.String }))
const Positioned = Schema.Struct({ position: Schema.Number })
const Titled = Schema.Array(Schema.Struct({ title: Schema.String }))

type Statement = readonly [sql: string, bindings: readonly SqliteBinding[]]
type DatabaseError = SqliteDriverError | SqliteWalRefused | SqliteCheckpointIncomplete

// Every store error is raised here, and logged once, with its cause.
const raise = <E extends ChatStoreError>(error: E): Effect.Effect<never, E> =>
  Effect.zipRight(Effect.logError("Chat store failed", Cause.fail(error)), Effect.fail(error))
const unavailable = (cause: DatabaseError | PlatformError) => raise(new ChatStoreUnavailable({ cause }))
const corrupt = (cause: ParseResult.ParseError) => raise(new ChatStoreCorrupt({ cause }))
// A value's type does not carry its schema's refinements, so encoding one can fail.
const rejected = (cause: ParseResult.ParseError) => raise(new ChatStoreRejected({ cause }))
const failed = (error: DatabaseError | ParseResult.ParseError): Effect.Effect<never, ChatStoreUnavailable | ChatStoreCorrupt> =>
  error._tag === "ParseError" ? corrupt(error) : unavailable(error)

const encodeEvent = (event: SessionEvent) => Effect.catchAll(Schema.encode(StoredEvent)(event), rejected)
const encodeRecord = (record: ChatSessionRecord) => Effect.catchAll(Schema.encode(StoredSession)(record), rejected)
const decodeEvents = Schema.decodeUnknown(Schema.Array(StoredEvent))
const decodePosition = Schema.decodeUnknown(Positioned)
const decodeTitled = Schema.decodeUnknown(Titled)
const decodeStoredSession = Schema.decodeUnknown(StoredSession, { onExcessProperty: "error" })
// A record that does not decode fails what reads it as corrupt.
const decodeRecord = (row: unknown) => Effect.catchAll(decodeStoredSession(row), corrupt)

// Creates a new file's schema, accepts a file of this format or an earlier one, upgrading it, and
// refuses any other file. The format is read and decided in the `BEGIN IMMEDIATE` transaction that
// applies the decision, so no other connection changes it in between: of two processes opening the
// same file, one applies the schema or upgrade and the other finds it applied.
const checkFormat = (db: SqliteConnection) => transaction(db, Effect.gen(function* () {
  const found = yield* userVersion(db)
  const present = yield* db.query("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").pipe(
    Effect.flatMap(Schema.decodeUnknown(Tables)),
    Effect.map((rows) => rows.map(({ name }) => name)),
  )
  const apply = (statements: ReadonlyArray<string>) => Effect.forEach([...statements, stamp], (sql) => db.execute(sql), { discard: true })
  if (found === 0 && present.length === 0) return yield* apply(schema)
  const recognised = Option.exists(Array.get(formats, found - 1), (expected) =>
    expected.length === present.length && expected.every((table) => present.includes(table)))
  if (!recognised) return yield* new ChatStoreIncompatible({ format: Option.some({ version: found, tables: present }) })
  if (found < chatStoreVersion) yield* apply(earlier.slice(found - 1).flatMap((upgrade) => upgrade.statements))
})).pipe(Effect.catchAll((error): Effect.Effect<never, ChatStoreError> => {
  switch (error._tag) {
    case "ChatStoreIncompatible":
      return raise(error)
    case "SqliteDriverNotADatabase":
      return raise(new ChatStoreIncompatible({ format: Option.none() }))
    default:
      return failed(error)
  }
}))

/** `ChatStore` over `<storage root>/chats.sqlite`. The file opens on first use; a failed open is retried by the next. */
export const ChatStoreLive = Layer.scoped(
  ChatStore,
  Effect.gen(function* () {
    const storage = yield* GlobalStorage
    const driver = yield* SqliteDriver
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    // One connection: a transaction spans several statements, so nothing else may use it meanwhile.
    const lock = yield* Effect.makeSemaphore(1)

    const connection = yield* ScopedCache.make({
      capacity: 1,
      lookup: (_: void) => Effect.gen(function* () {
        yield* fs.makeDirectory(storage.root, { recursive: true }).pipe(Effect.catchAll(unavailable))
        const db = yield* driver.open(path.join(storage.root, "chats.sqlite"), { create: true }).pipe(Effect.catchAll(unavailable))
        yield* db.execute("PRAGMA busy_timeout=5000").pipe(Effect.catchAll(unavailable))
        yield* checkFormat(db)
        yield* enableWal(db).pipe(Effect.catchAll(failed))
        yield* Effect.forEach(pragmas, (sql) => db.execute(sql), { discard: true }).pipe(Effect.catchAll(unavailable))
        return db
      }),
      timeToLive: Duration.infinity,
    })
    // An open that failed is forgotten, so the next call opens again.
    const opened = connection.get(undefined).pipe(Effect.tapError(() => connection.invalidate(undefined)))

    const use = <A, E>(f: (db: SqliteConnection) => Effect.Effect<A, E>) =>
      Effect.scoped(Effect.flatMap(opened, (db) => lock.withPermits(1)(f(db))))
    const query = (sql: string, bindings: readonly SqliteBinding[]) =>
      use((db) => Effect.catchAll(db.query(sql, bindings), unavailable))
    // Runs `f` in one transaction on the connection, its database errors raised as the store's.
    const atomically = <A>(f: (db: SqliteConnection) => Effect.Effect<A, DatabaseError | ParseResult.ParseError>) =>
      use((db) => Effect.catchAll(transaction(db, f(db)), failed))
    const write = (all: readonly Statement[]) =>
      atomically((db) => Effect.forEach(all, ([sql, bindings]) => db.execute(sql, bindings), { discard: true }))

    // Appends a commit's events, as statements that insert them.
    const appends = (id: SessionId, events: ReadonlyArray<SessionEvent>) =>
      Effect.forEach(events, (event) => Effect.map(encodeEvent(event), (row): Statement =>
        [append, [id, row.type, row.version, row.created_at, row.data]]))

    return ChatStore.of({
      session: (id: SessionId) => ({
        // The statements of `committedRecord`: the record is created first, then its caches apply.
        // Output alone changes neither cache.
        commit: (events) => Effect.gen(function* () {
          const created = yield* Effect.transposeMapOption(createdRecord(id, events), encodeRecord)
          const record = Option.match(created, {
            onNone: (): readonly Statement[] => [],
            onSome: (row): readonly Statement[] => [[create, [row.session_id, row.title, row.message_count, row.created_at, row.updated_at]]],
          })
          const cache = Option.match(cacheChange(events), {
            onNone: (): readonly Statement[] => [],
            onSome: ({ messages, updated }): readonly Statement[] => [[cached, [messages, DateTime.toEpochMillis(updated), id]]],
          })
          yield* write([...yield* appends(id, events), ...record, ...cache])
        }),
        read: Stream.paginateChunkEffect(0, (after) =>
          query(`SELECT position, type, version, created_at, data FROM events WHERE session_id = ? AND position > ? ORDER BY position LIMIT ${page}`, [id, after]).pipe(
            Effect.flatMap((rows) => Effect.all([
              decodeEvents(rows),
              rows.length < page
                ? Effect.succeed(Option.none<number>())
                : Effect.map(decodePosition(rows[rows.length - 1]), ({ position }) => Option.some(position)),
            ]).pipe(Effect.catchAll(corrupt))),
            Effect.map(([events, next]) => [Chunk.fromIterable(events), next] as const),
          )),
      }),
      record: (id) => query(`${recorded} WHERE session_id = ?`, [id]).pipe(
        Effect.flatMap((rows) => Effect.transposeMapOption(Array.head(rows), decodeRecord)),
      ),
      // The statement of `renamedRecord`.
      rename: (id, change) => atomically((db) => Effect.gen(function* () {
        const current = yield* Effect.flatMap(db.query(`SELECT title FROM sessions WHERE session_id = ?`, [id]), decodeTitled)
        const outcome = renameOutcome(Option.map(Array.head(current), ({ title }) => title), change)
        if (outcome === "renamed") {
          yield* db.execute("UPDATE sessions SET title = ?, updated_at = max(updated_at, ?) WHERE session_id = ?", [change.title, DateTime.toEpochMillis(change.time), id])
        }
        return outcome
      })),
      page: ({ offset, limit }) => Effect.flatMap(query(listed, [limit, offset]), (rows) => Effect.forEach(rows, decodeRecord)),
      delete: (id) => Effect.zipRight(
        write([
          ["DELETE FROM events WHERE session_id = ?", [id]],
          ["DELETE FROM sessions WHERE session_id = ?", [id]],
        ]),
        use((db) => Effect.catchAll(truncateWal(db), failed)),
      ),
    })
  })
)
