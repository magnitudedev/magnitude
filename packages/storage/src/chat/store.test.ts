import { Database } from "bun:sqlite"
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Cause, Chunk, DateTime, Effect, Either, Layer, Logger, LogLevel, Option, Stream } from "effect"
import {
  generateMessageId,
  generateSessionId,
  type MessageId,
  MessageId as MessageIdSchema,
  type MessageSent,
  type ResponseSettings,
  type ResponseStarted,
  ServingModelId,
  type SessionEvent,
  type SessionId,
} from "@magnitudedev/acn-protocol"
import { SqliteCheckpointIncomplete } from "../sqlite/database"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services/global-storage"
import { BunSqliteDriverLayer } from "../sqlite/bun"
import { ChatStore, type ChatSessionRecord, ChatStoreIncompatible, ChatStoreRejected, ChatStoreUnavailable } from "./store"
import { committedRecord, renamedRecord } from "./session-record"
import { ChatStoreLive } from "./store-live"

let root: string

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-chat-store-"))
})

afterEach(async () => {
  await rm(root, { recursive: true, force: true })
})

const live = (at = () => root) =>
  ChatStoreLive.pipe(
    Layer.provide(
      Layer.mergeAll(
        BunSqliteDriverLayer,
        BunFileSystem.layer,
        BunPath.layer,
        Layer.effect(GlobalStorage, Effect.sync(() => GlobalStorage.of({ root: at(), paths: makeGlobalStoragePaths(at()) })))
      )
    )
  )

const withStore = <A, E>(body: (store: ChatStore) => Effect.Effect<A, E>) =>
  Effect.runPromise(Effect.provide(Effect.flatMap(ChatStore, body), live()))

// A second connection to the same file, for inspecting and tampering with rows.
const raw = <A>(f: (db: Database) => A): A => {
  const db = new Database(join(root, "chats.sqlite"))
  try {
    return f(db)
  } finally {
    db.close()
  }
}

const base = 1_761_350_400_000
const at = (ms: number) => DateTime.unsafeMake(base + ms)
const newSession = () => Effect.runSync(generateSessionId)
const newMessage = () => Effect.runSync(generateMessageId)
const settings: ResponseSettings = {
  systemPrompt: "be brief",
  temperature: Option.some(0.5),
  maxOutputTokens: Option.none(),
  topP: Option.none(),
  topK: Option.some(40),
  minP: Option.none(),
  repetitionPenalty: Option.none(),
  presencePenalty: Option.none(),
  frequencyPenalty: Option.none(),
  seed: Option.none(),
  stop: ["\n"],
}
const model = ServingModelId.make("model-1")
const sent = (text: string, ms = 0): MessageSent => ({ _tag: "message_sent", time: at(ms), messageId: newMessage(), text })
const started = (ms = 0, replaces: Option.Option<MessageId> = Option.none()): ResponseStarted => ({
  _tag: "response_started", time: at(ms), responseId: newMessage(), replaces, model,
  reasoningEffort: Option.none(), settings,
})
const output = (ms: number, text = "out"): SessionEvent =>
  ({ _tag: "response_output", time: at(ms), responseId: newMessage(), part: "text", deltas: [text], at: [0] })
/** Creates a session whose first message and response are at `ms`. */
const create = (store: ChatStore, id: SessionId, text = "chat", ms = 0) =>
  store.session(id).commit([sent(text, ms), started(ms)])
const record = (id: SessionId, { title = "chat", created = 0, updated = 0, messageCount = 2 } = {}): ChatSessionRecord => ({
  id,
  title,
  time: { created: at(created), updated: at(updated) },
  messageCount,
})
const all = (store: ChatStore) => store.page({ offset: 0, limit: 1000 })

const readAll = (store: ChatStore, id: SessionId) => Effect.map(Stream.runCollect(store.session(id).read), Chunk.toArray)

describe("ChatStore format", () => {
  it("creates a new file's schema, and reopens its own", async () => {
    const id = newSession()
    await withStore((store) => create(store, id, "hi"))
    expect(raw((db) => db.query("PRAGMA user_version").get())).toEqual({ user_version: 1 })
    expect(await withStore((store) => readAll(store, id))).toHaveLength(2)
  })

  it("refuses main's chat file with ChatStoreIncompatible", async () => {
    raw((db) => {
      db.run("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
      db.run("CREATE TABLE messages (id TEXT PRIMARY KEY)")
      db.run("CREATE TABLE generations (id TEXT PRIMARY KEY)")
      db.run("PRAGMA user_version=1")
    })
    const failure = await withStore((store) => Effect.flip(all(store)))
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.some({ version: 1, tables: ["conversations", "generations", "messages"] }) }))
    expect(failure._tag === "ChatStoreIncompatible" && failure.newer).toBe(false)
  })

  it("refuses a file of another version", async () => {
    raw((db) => {
      db.run("CREATE TABLE events (position INTEGER PRIMARY KEY)")
      db.run("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
      db.run("PRAGMA user_version=2")
    })
    const failure = await withStore((store) => Effect.flip(all(store)))
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.some({ version: 2, tables: ["events", "sessions"] }) }))
    expect(failure._tag === "ChatStoreIncompatible" && failure.newer).toBe(true)
  })
})

describe("ChatStore format decisions", () => {
  it("decides the format in the transaction that applies it, so a process that created it meanwhile wins cleanly", async () => {
    // Another process creates this format's tables in a write transaction it holds briefly.
    const creator = Bun.spawn(["bun", "-e", `
      const { Database } = require("bun:sqlite")
      const db = new Database(${JSON.stringify(join(root, "chats.sqlite"))})
      db.run("BEGIN IMMEDIATE")
      db.run("CREATE TABLE events (position INTEGER PRIMARY KEY)")
      db.run("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL, message_count INTEGER NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)")
      db.run("PRAGMA user_version=1")
      console.log("locked")
      Bun.sleepSync(300)
      db.run("COMMIT")
    `], { stdout: "pipe" })
    const output = creator.stdout.getReader()
    const { value } = await output.read()
    expect(new TextDecoder().decode(value)).toContain("locked")
    // The file still reads as new until the creator commits.
    expect(await withStore((store) => all(store))).toEqual([])
    expect(await creator.exited).toBe(0)
  })
})

describe("ChatStore layout", () => {
  const message = MessageIdSchema.make("message-1")
  const response = MessageIdSchema.make("response-1")
  const first: MessageSent = { _tag: "message_sent", time: at(1), messageId: message, text: "  hi\n there " }
  const reply: ResponseStarted = {
    _tag: "response_started", time: at(2), responseId: response, replaces: Option.none(), model,
    reasoningEffort: Option.some("high"), settings,
  }
  const rest: ReadonlyArray<SessionEvent> = [
    { _tag: "response_output", time: at(3), responseId: response, part: "text", deltas: ["he", "llo"], at: [0, 7] },
    {
      _tag: "response_ended", time: at(4), responseId: response, outcome: { _tag: "finished", finishReason: "stop" },
      usage: Option.some({ promptTokens: 3, cachedTokens: 1, completionTokens: 2, draftTokens: Option.none(), draftAcceptedTokens: Option.none() }),
      timings: Option.some({ promptMs: 4, predictedMs: 5, samplerMs: 0.5, parserMs: 0.25, timeToFirstTokenMs: 6 }),
    },
  ]
  const storedSettings = { systemPrompt: "be brief", temperature: 0.5, topK: 40, stop: ["\n"] }

  it("stores each event type and a record in exactly this row layout", async () => {
    const id = newSession()
    await withStore((store) => Effect.zipRight(store.session(id).commit([first, reply]), store.session(id).commit(rest)))
    const rows = raw((db) => db.query("SELECT * FROM events ORDER BY position").all()) as Array<{ readonly data: string }>
    expect(rows.map(({ data, ...columns }) => ({ ...columns, data: JSON.parse(data) }))).toStrictEqual([
      { position: 1, session_id: id, type: "message_sent", version: 1, created_at: base + 1, data: { messageId: message, text: "  hi\n there " } },
      {
        position: 2, session_id: id, type: "response_started", version: 1, created_at: base + 2,
        data: { responseId: response, model, reasoningEffort: "high", settings: storedSettings },
      },
      {
        position: 3, session_id: id, type: "response_output", version: 1, created_at: base + 3,
        data: { responseId: response, part: "text", deltas: ["he", "llo"], at: [0, 7] },
      },
      {
        position: 4, session_id: id, type: "response_ended", version: 1, created_at: base + 4,
        data: {
          responseId: response, outcome: { type: "finished", finishReason: "stop" },
          usage: { promptTokens: 3, cachedTokens: 1, completionTokens: 2 },
          timings: { promptMs: 4, predictedMs: 5, samplerMs: 0.5, parserMs: 0.25, timeToFirstTokenMs: 6 },
        },
      },
    ])
    // The title is the first message's text, whitespace collapsed.
    expect(raw((db) => db.query("SELECT * FROM sessions").all())).toStrictEqual([{
      session_id: id,
      title: "hi there",
      message_count: 2,
      created_at: base + 1,
      updated_at: base + 4,
    }])
    expect(await withStore((store) => readAll(store, id))).toEqual([first, reply, ...rest])
  })

  it("refuses event JSON that repeats a value its row's columns own", async () => {
    const owned = ["type", "version", "time"] as const
    const ids = owned.map(newSession)
    await withStore((store) => Effect.forEach(ids, (id) => create(store, id)))
    raw((db) => owned.forEach((key, n) =>
      db.run(`UPDATE events SET data = json_set(data, '$.${key}', ${key === "type" ? "type" : key === "version" ? "version" : "created_at"}) WHERE session_id = ?`, [ids[n]])))
    const reads = await withStore((store) => Effect.forEach(ids, (id) => Effect.flip(readAll(store, id))))
    expect(reads.map(({ _tag }) => _tag)).toEqual(["ChatStoreCorrupt", "ChatStoreCorrupt", "ChatStoreCorrupt"])
  })

  it("rejects a value that breaks its schema's refinements, and writes nothing", async () => {
    const id = newSession()
    await withStore((store) => create(store, id))
    // A wall-clock step back gives a negative arrival offset.
    const stepped = { ...output(5), at: [0, -4], deltas: ["a", "b"] } as SessionEvent
    const failure = await withStore((store) => Effect.flip(store.session(id).commit([sent("more", 4), stepped])))
    expect(failure._tag).toBe("ChatStoreRejected")
    expect(failure).toBeInstanceOf(ChatStoreRejected)
    expect(await withStore((store) => Effect.all([Effect.map(readAll(store, id), (events) => events.length), store.record(id)])))
      .toEqual([2, Option.some(record(id))])
  })
})

describe("ChatStore sessions", () => {
  it("reads a session's events in commit order, across pages, apart from other sessions", async () => {
    const a = newSession()
    const b = newSession()
    const events = Array.from({ length: 1003 }, (_, n) => sent(`m${n}`, n))
    await withStore((store) => Effect.gen(function* () {
      yield* create(store, a, "a")
      yield* create(store, b, "b")
      yield* store.session(a).commit(events.slice(0, 500))
      yield* store.session(b).commit([sent("other")])
      yield* store.session(a).commit(events.slice(500))
    }))
    const [readA, readB] = await withStore((store) => Effect.all([readAll(store, a), readAll(store, b)]))
    expect(readA.slice(2)).toEqual(events)
    expect(readB.slice(2).map((event) => event._tag === "message_sent" && event.text)).toEqual(["other"])
  })

  it("creates the record with the first commit and keeps its caches of the log with each commit, as committedRecord does", async () => {
    const id = newSession()
    const commits: ReadonlyArray<ReadonlyArray<SessionEvent>> = [
      [sent("chat", 1), started(1)],
      // Output changes neither cache.
      [output(9)],
      // A retry replaces a response: it is no new message, and it is observed.
      [started(10, Option.some(newMessage())), output(11)],
      // A later message leaves the title as the first one set it.
      [sent("next", 12), started(12), output(20)],
    ]
    const records = await withStore((store) => Effect.forEach(commits, (events) =>
      Effect.zipRight(store.session(id).commit(events), store.record(id))))
    const expected = [
      record(id, { created: 1, updated: 1, messageCount: 2 }),
      record(id, { created: 1, updated: 1, messageCount: 2 }),
      record(id, { created: 1, updated: 10, messageCount: 2 }),
      record(id, { created: 1, updated: 12, messageCount: 4 }),
    ].map(Option.some)
    expect(records).toEqual(expected)
    // The rule a holder of the record in memory applies gives the same records.
    const folded = commits.reduce<ReadonlyArray<Option.Option<ChatSessionRecord>>>(
      (all, events) => [...all, committedRecord(id, all.at(-1) ?? Option.none(), events)], [])
    expect(folded).toEqual(expected)
  })

  it("commits events and the caches together or not at all", async () => {
    const id = newSession()
    await withStore((store) => create(store, id, "first"))
    raw((db) => db.run("CREATE TRIGGER refuse BEFORE UPDATE ON sessions BEGIN SELECT RAISE(ABORT, 'refused'); END"))
    const failure = await withStore((store) => Effect.flip(store.session(id).commit([sent("second", 2)])))
    expect(failure._tag).toBe("ChatStoreUnavailable")
    const [events, stored] = await withStore((store) => Effect.all([readAll(store, id), store.record(id)]))
    expect(events.map((event) => event._tag === "message_sent" && event.text)).toEqual(["first", false])
    expect(stored).toEqual(Option.some(record(id, { title: "first" })))
  })

  it("creates a session's record with the commit of its first message, together or not at all", async () => {
    const id = newSession()
    // The first use creates the file's schema.
    await withStore((store) => all(store))
    raw((db) => db.run("CREATE TRIGGER refuse BEFORE INSERT ON events WHEN NEW.type = 'response_started' BEGIN SELECT RAISE(ABORT, 'refused'); END"))
    expect((await withStore((store) => Effect.flip(create(store, id))))._tag).toBe("ChatStoreUnavailable")
    expect(await withStore((store) => Effect.all([readAll(store, id), store.record(id)]))).toEqual([[], Option.none()])
  })

  it("renames as a compare-and-set against the current title, touching the latest observation", async () => {
    const id = newSession()
    const missing = newSession()
    const outcomes = await withStore((store) => Effect.gen(function* () {
      yield* create(store, id, "first")
      return [
        yield* store.rename(id, { from: "first", title: "second", time: at(7) }),
        // A replay finds the title already set.
        yield* store.rename(id, { from: "first", title: "second", time: at(8) }),
        // A rename from a title the session no longer has is refused.
        yield* store.rename(id, { from: "first", title: "third", time: at(9) }),
        yield* store.rename(missing, { from: "first", title: "second", time: at(9) }),
      ]
    }))
    expect(outcomes).toEqual(["renamed", "unchanged", "title_changed", "missing"])
    const renamed = await withStore((store) => store.record(id))
    expect(renamed).toEqual(Option.some(record(id, { title: "second", updated: 7 })))
    expect(renamed).toEqual(Option.some(renamedRecord(record(id, { title: "first" }), { title: "second", time: at(7) })))
    // A later event older than the rename's time leaves the latest observation where it is.
    await withStore((store) => store.session(id).commit([sent("late", 5)]))
    const after = await withStore((store) => store.record(id))
    expect(after).toEqual(Option.some(record(id, { title: "second", updated: 7, messageCount: 3 })))
    expect(await withStore((store) => Effect.all([store.record(missing), all(store)]))).toEqual([Option.none(), [Option.getOrThrow(after)]])
  })

  it("pages records by latest update, then session id", async () => {
    const ids = [newSession(), newSession(), newSession()].sort()
    await withStore((store) => Effect.forEach([[ids[2], 5], [ids[0], 9], [ids[1], 5]] as const, ([id, updated]) =>
      create(store, id, "chat", updated)))
    const [first, rest] = await withStore((store) => Effect.all([store.page({ offset: 0, limit: 2 }), store.page({ offset: 2, limit: 2 })]))
    expect([...first, ...rest].map((row) => row.id)).toEqual([ids[0], ids[1], ids[2]])
  })

  it("decodes a record strictly, failing a page or a record that does not decode as corrupt", async () => {
    const good = newSession()
    const negative = newSession()
    const untimed = newSession()
    const untitled = newSession()
    await withStore((store) => Effect.forEach([good, negative, untimed, untitled], (id) => create(store, id)))
    raw((db) => {
      db.run("UPDATE sessions SET message_count = -1 WHERE session_id = ?", [negative])
      db.run("UPDATE sessions SET updated_at = 'later' WHERE session_id = ?", [untimed])
      db.run("UPDATE sessions SET title = x'6869' WHERE session_id = ?", [untitled])
      db.run("INSERT INTO sessions (session_id, title, message_count, created_at, updated_at) VALUES ('', 'chat', 1, 0, 0)")
    })
    // Each broken row fails the page that holds it: of the five one-row pages, only the good row's lists.
    const pages = await withStore((store) => Effect.forEach([0, 1, 2, 3, 4], (offset) => Effect.either(store.page({ offset, limit: 1 }))))
    expect(pages.filter(Either.isRight)).toEqual([Either.right([record(good)])])
    expect(pages.filter(Either.isLeft).map((page) => page.left._tag)).toEqual(Array(4).fill("ChatStoreCorrupt"))
    const [stored, broken] = await withStore((store) => Effect.all([store.record(good), Effect.flip(store.record(negative))]))
    expect(stored).toEqual(Option.some(record(good)))
    expect(broken._tag).toBe("ChatStoreCorrupt")
  })

  it("titles a session by its first message: whitespace collapsed, at most 80 code points", async () => {
    const id = newSession()
    await withStore((store) => create(store, id, ` ${"😀".repeat(90)}\n`))
    const stored = Option.getOrThrow(await withStore((store) => store.record(id)))
    expect(stored.title).toBe("😀".repeat(80))
  })

  it("has no record for a session that does not exist", async () => {
    expect(await withStore((store) => store.record(newSession()))).toEqual(Option.none())
  })

  it("fails a read whose event does not decode", async () => {
    const id = newSession()
    await withStore((store) => create(store, id))
    raw((db) => db.run("UPDATE events SET version = 9"))
    expect((await withStore((store) => Effect.flip(readAll(store, id))))._tag).toBe("ChatStoreCorrupt")
  })

  it("deletes a session's rows from both tables, then truncates the write-ahead log", async () => {
    const gone = newSession()
    const kept = newSession()
    const wal = await withStore((store) => Effect.gen(function* () {
      for (const id of [gone, kept]) yield* create(store, id, "secret")
      yield* store.delete(gone)
      return yield* Effect.promise(() => stat(join(root, "chats.sqlite-wal")))
    }))
    expect(wal.size).toBe(0)
    expect(raw((db) => db.query("SELECT DISTINCT session_id FROM events").all())).toEqual([{ session_id: kept }])
    expect(raw((db) => db.query("SELECT session_id FROM sessions").all())).toEqual([{ session_id: kept }])
  })
})

describe("ChatStore delete", () => {
  it("fails while a reader blocks the write-ahead log's truncation, and a retry finishes it", async () => {
    const gone = newSession()
    const kept = newSession()
    await withStore((store) => Effect.forEach([[gone, "gone-secret"], [kept, "kept"]] as const, ([id, text]) => create(store, id, text)))
    const reader = new Database(join(root, "chats.sqlite"))
    const [blocked, retried] = await withStore((store) => Effect.gen(function* () {
      // The reader's snapshot holds the log's frames.
      reader.run("BEGIN")
      reader.query("SELECT COUNT(*) FROM events").get()
      const blocked = yield* Effect.flip(store.delete(gone))
      reader.run("COMMIT")
      reader.close()
      const retried = yield* Effect.either(store.delete(gone))
      return [blocked, retried] as const
    }))
    expect(blocked).toBeInstanceOf(ChatStoreUnavailable)
    expect(blocked._tag === "ChatStoreUnavailable" && blocked.cause).toBeInstanceOf(SqliteCheckpointIncomplete)
    expect(retried).toEqual(Either.right(undefined))
    expect((await stat(join(root, "chats.sqlite-wal"))).size).toBe(0)
    expect((await readFile(join(root, "chats.sqlite"))).includes("gone-secret")).toBe(false)
    expect(raw((db) => db.query("SELECT session_id FROM sessions").all())).toEqual([{ session_id: kept }])
  }, 15_000)
})

describe("ChatStore errors", () => {
  it("logs each error once, with its cause, where the store raises it", async () => {
    await writeFile(join(root, "chats.sqlite"), "not a database ".repeat(40))
    const logged: Array<{ readonly level: LogLevel.LogLevel; readonly cause: Cause.Cause<unknown> }> = []
    const capture = Logger.make(({ logLevel, cause }) => { logged.push({ level: logLevel, cause }) })
    const failure = await Effect.runPromise(Effect.provide(Effect.flatMap(ChatStore, (store) => Effect.flip(all(store))), Layer.merge(
      live(),
      Logger.replace(Logger.defaultLogger, capture),
    )))
    expect(logged).toEqual([{ level: LogLevel.Error, cause: Cause.fail(failure) }])
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.none() }))
    expect(Cause.pretty(logged[0].cause)).toContain("Not a chat store: not a SQLite database")
  })

  it("says in its message which format an incompatible file has", () => {
    expect(new ChatStoreIncompatible({ format: Option.some({ version: 7, tables: ["events", "sessions"] }) }).message)
      .toBe("Not a chat store of this format or an earlier one: version 7, tables events, sessions")
    expect(new ChatStoreIncompatible({ format: Option.some({ version: 0, tables: [] }) }).message)
      .toBe("Not a chat store of this format or an earlier one: version 0, tables none")
  })
})

describe("ChatStore connection", () => {
  it("opens nothing when built, and retries a failed open on the next call", async () => {
    // The storage root is a file, so the first open cannot create its directory.
    const blocked = join(root, "blocked")
    await writeFile(blocked, "")
    const results = await Effect.runPromise(Effect.provide(Effect.flatMap(ChatStore, (store) => Effect.gen(function* () {
      const first = yield* Effect.either(all(store))
      yield* Effect.promise(() => rm(blocked))
      const second = yield* Effect.either(all(store))
      return [first, second] as const
    })), live(() => blocked)))
    expect(Either.isLeft(results[0]) && results[0].left._tag).toBe("ChatStoreUnavailable")
    expect(results[1]).toEqual(Either.right([]))
  })
})
