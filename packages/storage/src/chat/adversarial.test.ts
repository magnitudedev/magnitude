import { Database } from "bun:sqlite"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Cause, Chunk, DateTime, Effect, Exit, Fiber, Layer, Logger, Option, Stream } from "effect"
import {
  generateMessageId,
  generateSessionId,
  type MessageSent,
  ServingModelId,
  type SessionEvent,
  type SessionId,
} from "@magnitudedev/acn-protocol"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services/global-storage"
import { BunSqliteDriverLayer } from "../sqlite/bun"
import { ChatStore, type ChatSessionRecord, ChatStoreIncompatible } from "./store"
import { ChatStoreLive } from "./store-live"

let root: string

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "magnitude-chat-adversarial-"))
})

afterEach(async () => {
  await rm(root, { recursive: true, force: true })
})

const live = () =>
  ChatStoreLive.pipe(
    Layer.provide(
      Layer.mergeAll(
        BunSqliteDriverLayer,
        BunFileSystem.layer,
        BunPath.layer,
        Layer.effect(GlobalStorage, Effect.sync(() => GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) }))),
      ),
    ),
  )

const withStore = <A, E>(body: (store: ChatStore) => Effect.Effect<A, E>) =>
  Effect.runPromise(Effect.provide(Effect.flatMap(ChatStore, body), live()))

/** Runs against the store, with every cause it logs. */
const withLoggedStore = async <A, E>(body: (store: ChatStore) => Effect.Effect<A, E>) => {
  const logged: Array<Cause.Cause<unknown>> = []
  const capture = Logger.make(({ cause }) => { logged.push(cause) })
  const result = await Effect.runPromise(Effect.provide(Effect.flatMap(ChatStore, body), Layer.merge(live(), Logger.replace(Logger.defaultLogger, capture))))
  return [result, logged] as const
}

const raw = <A>(f: (db: Database) => A): A => {
  const db = new Database(join(root, "chats.sqlite"))
  try {
    return f(db)
  } finally {
    db.close()
  }
}

const at = (ms: number) => DateTime.unsafeMake(1_761_350_400_000 + ms)
const newSession = () => Effect.runSync(generateSessionId)
const newMessage = () => Effect.runSync(generateMessageId)
const sent = (text: string, ms = 0): MessageSent => ({ _tag: "message_sent", time: at(ms), messageId: newMessage(), text })
const settings = {
  systemPrompt: "",
  temperature: Option.none(),
  maxOutputTokens: Option.none(),
  topP: Option.none(),
  topK: Option.none(),
  minP: Option.none(),
  repetitionPenalty: Option.none(),
  presencePenalty: Option.none(),
  frequencyPenalty: Option.none(),
  seed: Option.none(),
  stop: [],
}
/** Creates a session whose first message and response are at `ms`. */
const create = (store: ChatStore, id: SessionId, text = "chat", ms = 0) => store.session(id).commit([sent(text, ms), {
  _tag: "response_started", time: at(ms), responseId: newMessage(), replaces: Option.none(),
  model: ServingModelId.make("model"), reasoningEffort: Option.none(), settings,
}])
const record = (id: SessionId, { title = "chat", created = 0, updated = 0, messageCount = 2 } = {}): ChatSessionRecord => ({
  id,
  title,
  time: { created: at(created), updated: at(updated) },
  messageCount,
})
const all = (store: ChatStore) => store.page({ offset: 0, limit: 1000 })
const readAll = (store: ChatStore, id: SessionId) => Effect.map(Stream.runCollect(store.session(id).read), Chunk.toArray)

describe("adversarial: chat store writes", () => {
  it("keeps every session's commit order under concurrent commits across 40 sessions", async () => {
    const ids = Array.from({ length: 40 }, newSession)
    const result = await withStore((store) => Effect.gen(function* () {
      yield* Effect.forEach(ids, (id) => Effect.zipRight(create(store, id), Effect.forEach(Array.from({ length: 15 }, (_, n) => n), (n) =>
        store.session(id).commit([sent(`${id}:${n}`, n)]))), { concurrency: "unbounded", discard: true })
      return yield* Effect.forEach(ids, (id) => readAll(store, id))
    }))
    result.forEach((events, i) => {
      expect(events.slice(2).map((e) => e._tag === "message_sent" && e.text)).toEqual(Array.from({ length: 15 }, (_, n) => `${ids[i]}:${n}`))
    })
    const sessions = await withStore((store) => all(store))
    expect(sessions).toHaveLength(40)
    for (const s of sessions) expect(s).toEqual(record(s.id, { updated: 14, messageCount: 17 }))
  })

  it("rolls back every event and the caches when a later statement of the commit fails", async () => {
    const id = newSession()
    await withStore((store) => create(store, id, "first"))
    raw((db) => db.run("CREATE TRIGGER refuse BEFORE INSERT ON events WHEN NEW.data LIKE '%poison%' BEGIN SELECT RAISE(ABORT, 'refused'); END"))
    const failure = await withStore((store) => Effect.flip(store.session(id).commit(
      [sent("ok-1", 9), sent("ok-2", 9), sent("poison", 9), sent("ok-3", 9)],
    )))
    expect(failure._tag).toBe("ChatStoreUnavailable")
    const [events, sessions] = await withStore((store) => Effect.all([readAll(store, id), all(store)]))
    expect(events.map((e) => e._tag === "message_sent" && e.text)).toEqual(["first", false])
    expect(sessions).toEqual([record(id, { title: "first" })])
    // The connection is usable after the failed transaction.
    await withStore((store) => store.session(id).commit([sent("after")]))
    expect((await withStore((store) => readAll(store, id))).length).toBe(3)
  })

  it("an interrupted commit is all or nothing", async () => {
    const id = newSession()
    const events = Array.from({ length: 4000 }, (_, n) => sent(`m${n}`, n))
    await withStore((store) => Effect.gen(function* () {
      yield* create(store, id)
      const fiber = yield* Effect.fork(store.session(id).commit(events))
      yield* Effect.sleep("1 millis")
      yield* Fiber.interrupt(fiber)
    }))
    const [read, stored] = await withStore((store) => Effect.all([readAll(store, id), store.record(id)]))
    expect([2, 4002]).toContain(read.length)
    // The caches agree with whichever log the commit left.
    expect(stored).toEqual(Option.some(record(id, read.length === 2 ? {} : { updated: 3999, messageCount: 4002 })))
  })

  it("stores and reads back very large rows", async () => {
    const id = newSession()
    const big = "x".repeat(16 * 1024 * 1024)
    const output: SessionEvent = { _tag: "response_output", time: at(1), responseId: newMessage(), part: "text", deltas: [big, "y"], at: [0, 5] }
    const read = await withStore((store) => Effect.zipRight(
      Effect.zipRight(create(store, id, `${"t".repeat(100_000)} ${big}`), store.session(id).commit([output])),
      Effect.all([readAll(store, id), all(store)]),
    ))
    expect(read[0][0]).toEqual(expect.objectContaining({ text: `${"t".repeat(100_000)} ${big}` }))
    expect(read[0][2]).toEqual(output)
    // The title is the first 80 code points of the text.
    expect(read[1]).toEqual([record(id, { title: "t".repeat(80) })])
  }, 30_000)
})

describe("adversarial: chat store reads", () => {
  it("a delete during an open read stream ends the stream without an error and leaves nothing", async () => {
    const id = newSession()
    const events = Array.from({ length: 2500 }, (_, n) => sent(`m${n}`, n))
    const read = await withStore((store) => Effect.gen(function* () {
      yield* create(store, id)
      yield* store.session(id).commit(events)
      let deleted = false
      return yield* Stream.runCollect(store.session(id).read.pipe(
        Stream.tap(() => deleted ? Effect.void : Effect.zipRight(store.delete(id), Effect.sync(() => { deleted = true }))),
      ))
    }))
    // The page already read is all the read returns: a prefix of the log, in order.
    const texts = Chunk.toArray(read).slice(2).map((e) => e._tag === "message_sent" && e.text)
    expect(texts).toEqual(events.slice(0, texts.length).map((e) => e.text))
    expect(texts.length).toBe(998)
    expect(raw((db) => db.query("SELECT COUNT(*) AS n FROM events").get())).toEqual({ n: 0 })
    expect(raw((db) => db.query("SELECT COUNT(*) AS n FROM sessions").get())).toEqual({ n: 0 })
  })

  it("pages sessions with equal updated_at values deterministically, with no duplicates or gaps", async () => {
    const ids = Array.from({ length: 31 }, newSession)
    await withStore((store) => Effect.forEach(ids, (id, n) =>
      create(store, id, "chat", n % 3 === 0 ? 7 : 3), { concurrency: "unbounded", discard: true }))
    const pages = await withStore((store) => Effect.forEach([0, 7, 14, 21, 28, 35], (offset) => store.page({ offset, limit: 7 })))
    const listed = pages.flat().map((row) => row.id)
    const expected = [
      ...ids.filter((_, n) => n % 3 === 0).sort(),
      ...ids.filter((_, n) => n % 3 !== 0).sort(),
    ]
    expect(listed).toEqual(expected)
    expect(pages.at(-1)).toEqual([])
  })

  it("fails a page holding a row whose id is valid but whose count is not a count as corrupt, logged once", async () => {
    const good = newSession()
    const bad = newSession()
    await withStore((store) => Effect.forEach([good, bad], (id) => create(store, id, "chat", 1)))
    raw((db) => db.run("UPDATE sessions SET message_count = 1.5 WHERE session_id = ?", [bad]))
    const [failure, logged] = await withLoggedStore((store) => Effect.flip(store.page({ offset: 0, limit: 10 })))
    expect(failure._tag).toBe("ChatStoreCorrupt")
    expect(logged).toEqual([Cause.fail(failure)])
  })

  it("fails only the pages holding a row whose session id does not decode, in its place in the list order", async () => {
    const good = newSession()
    await withStore((store) => create(store, good, "chat", 1))
    raw((db) => db.run("INSERT INTO sessions (session_id, title, message_count, created_at, updated_at) VALUES ('', 'chat', 1, 0, 0)"))
    const [[before, failure], logged] = await withLoggedStore((store) =>
      Effect.all([store.page({ offset: 0, limit: 1 }), Effect.flip(all(store))]))
    expect(before).toEqual([record(good, { created: 1, updated: 1 })])
    expect(failure._tag).toBe("ChatStoreCorrupt")
    expect(logged).toEqual([Cause.fail(failure)])
  })
})

describe("adversarial: chat store file format", () => {
  it("creates the schema in an empty (zero-byte) file", async () => {
    await writeFile(join(root, "chats.sqlite"), "")
    const id = newSession()
    await withStore((store) => create(store, id, "hi"))
    expect((await withStore((store) => readAll(store, id))).length).toBe(2)
  })

  it("refuses a file that is not SQLite as incompatible, and leaves it untouched", async () => {
    const foreign = "this is not a sqlite database, it is a text file that a user put here. ".repeat(20)
    await writeFile(join(root, "chats.sqlite"), foreign)
    const failure = await withStore((store) => Effect.flip(all(store)))
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.none() }))
    expect(failure._tag === "ChatStoreIncompatible" && failure.newer).toBe(false)
    expect(await Bun.file(join(root, "chats.sqlite")).text()).toBe(foreign)
  })

  it("refuses a newer user_version with our exact tables as newer, and writes nothing", async () => {
    raw((db) => {
      db.run("CREATE TABLE events (position INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, type TEXT NOT NULL, version INTEGER NOT NULL, created_at INTEGER NOT NULL, data TEXT NOT NULL)")
      db.run("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL, message_count INTEGER NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)")
      db.run("PRAGMA user_version=7")
    })
    const id = newSession()
    const failure = await withStore((store) => Effect.flip(create(store, id, "x")))
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.some({ version: 7, tables: ["events", "sessions"] }) }))
    expect(failure._tag === "ChatStoreIncompatible" && failure.newer).toBe(true)
    expect(raw((db) => db.query("SELECT COUNT(*) AS n FROM events").get())).toEqual({ n: 0 })
  })

  it("refuses version 0 with tables present (a half-created or foreign file)", async () => {
    raw((db) => db.run("CREATE TABLE events (x)"))
    const failure = await withStore((store) => Effect.flip(all(store)))
    expect(failure).toEqual(new ChatStoreIncompatible({ format: Option.some({ version: 0, tables: ["events"] }) }))
  })

  it("keeps working when another process opens the same file meanwhile", async () => {
    const id = newSession()
    const exit = await Effect.runPromiseExit(Effect.provide(Effect.flatMap(ChatStore, (store) => Effect.gen(function* () {
      yield* create(store, id, "a")
      raw((db) => db.query("SELECT COUNT(*) FROM events").get())
      yield* store.session(id).commit([sent("b")])
      return yield* readAll(store, id)
    })), live()))
    expect(Exit.isSuccess(exit) && exit.value.length).toBe(3)
  })
})
