import { afterEach, expect, it, vi } from "vitest"
import { Effect, Fiber, Layer, Option, Stream } from "effect"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { GlobalStorage, makeGlobalStorage } from "@magnitudedev/storage"
import { ServingModelId, type ServingUsageRequest } from "@magnitudedev/acn-protocol"
import { ServingUsage, ServingUsageLive, ServingUsageRecord } from "./serving-usage"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
const roots: string[] = []
afterEach(() => { vi.useRealTimers(); roots.splice(0).forEach(root => rmSync(root, { recursive: true, force: true })) })
const root = () => { const value = mkdtempSync(join(tmpdir(), "magnitude-usage-test-")); roots.push(value); return value }
const layer = (root: string) => ServingUsageLive.pipe(Layer.provide(Layer.mergeAll(BunFileSystem.layer, BunPath.layer, Layer.succeed(GlobalStorage, makeGlobalStorage({ root })))))
const query: ServingUsageRequest = { period: "AllTime", timeZone: "America/Los_Angeles", model: Option.none() }
const record = (id: string, overrides: Partial<ServingUsageRecord> = {}): ServingUsageRecord => ({ id: ServingUsageRecord.fields.id.make(id), model: ServingModelId.make("model-a"), completedAt: Date.now(), input: 100, cached: 40, output: 20, generationMs: 1000, firstTokenMs: 200, complete: true, drafter: null, draftTokens: null, acceptedDraftTokens: null, ...overrides })
it("persists once across restarts and filters models without adding cached input twice", async () => {
  const directory = root()
  await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("one"))
    yield* store.record(record("one"))
    yield* store.record(record("two", { model: ServingModelId.make("model-b"), output: 80, generationMs: 2000, firstTokenMs: 400 }))
  }).pipe(Effect.provide(layer(directory))))
  const values = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    return [yield* store.read(query), yield* store.read({ ...query, model: Option.some(ServingModelId.make("model-b")) })]
  }).pipe(Effect.provide(layer(directory))))
  expect(values[0]).toMatchObject({ _tag: "Available", requests: 2, inputTokens: 200, cachedInputTokens: 80, outputTokens: 100, totalTokens: 300, tokensPerSecond: 100 / 3, timeToFirstTokenMs: 300 })
  expect(values[1]).toMatchObject({ requests: 1, totalTokens: 180, tokensPerSecond: 40, models: [{ id: "model-a", requests: 1 }, { id: "model-b", requests: 1 }] })
})
it("uses the caller's local day across a DST boundary", async () => {
  vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-03-08T18:00:00Z"))
  const value = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("yesterday", { completedAt: Date.parse("2026-03-08T07:59:59Z") }))
    yield* store.record(record("today", { completedAt: Date.parse("2026-03-08T08:00:00Z") }))
    return yield* store.read({ ...query, period: "Today" })
  }).pipe(Effect.provide(layer(root()))))
  vi.restoreAllMocks()
  expect(value).toMatchObject({ requests: 1, totalTokens: 120 })
})
it("excludes absent timing samples and reports partial evidence", async () => {
  const value = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("known"))
    yield* store.record(record("partial", { input: 5, cached: null, output: null, generationMs: null, firstTokenMs: null, complete: false }))
    return yield* store.read(query)
  }).pipe(Effect.provide(layer(root()))))
  expect(value).toMatchObject({ requests: 2, incompleteRequests: 1, totalTokens: 125, cachedInputRequests: 1, speedSamples: 1, latencySamples: 1, tokensPerSecond: 20, timeToFirstTokenMs: 200 })
})
it("reports damaged storage as unavailable rather than empty history and does not fail serving writes", async () => {
  const directory = root(); writeFileSync(join(directory, "serving-usage.sqlite"), "corrupt")
  const value = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("failed"))
    return yield* store.read(query)
  }).pipe(Effect.provide(layer(directory))))
  expect(value._tag).toBe("Unavailable")
})
it("starts with real empty totals and unavailable timing averages", async () => {
  const value = await Effect.runPromise(Effect.flatMap(ServingUsage, store => store.read(query)).pipe(Effect.provide(layer(root()))))
  expect(value).toMatchObject({ requests: 0, totalTokens: 0, tokensPerSecond: null, timeToFirstTokenMs: null, models: [] })
})

it("buckets a full calendar independently of Today and counts cached input once", async () => {
  vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-03-09T18:00:00Z"))
  const values = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    for (const [id, timestamp] of [["before", "2026-03-08T07:59:59Z"], ["start", "2026-03-08T08:00:00Z"], ["end", "2026-03-09T06:59:59Z"], ["next", "2026-03-09T07:00:00Z"]]) {
      yield* store.record(record(id!, { completedAt: Date.parse(timestamp!) }))
    }
    yield* store.record(record("other", { model: ServingModelId.make("model-b"), output: 80 }))
    yield* store.record(record("old", { completedAt: Date.parse("2020-01-01T00:00:00Z") }))
    return [yield* store.read({ ...query, period: "Today" }), yield* store.read({ ...query, model: Option.some(ServingModelId.make("model-a")) })]
  }).pipe(Effect.provide(layer(root()))))
  vi.restoreAllMocks()
  const [all, selected] = values
  if (all?._tag !== "Available" || selected?._tag !== "Available") throw new Error("Missing activity")
  expect(all.totalTokens).toBe(300)
  expect(all.dailyActivity.slice(-3)).toEqual([
    { date: "2026-03-07", totalTokens: 120 }, { date: "2026-03-08", totalTokens: 240 }, { date: "2026-03-09", totalTokens: 300 },
  ])
  expect(selected.dailyActivity.at(-1)?.totalTokens).toBe(120)
  expect(all.dailyActivity.length).toBe(366)
  expect(new Date(`${all.dailyActivity[0]!.date}T00:00:00Z`).getUTCDay()).toBe(0)
  expect(all.dailyActivity.reduce((sum, day) => sum + day.totalTokens, 0)).toBe(660)
})

it("handles fractional-offset midnight and the repeated hour at fall DST", async () => {
  vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-11-02T18:00:00Z"))
  const values = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    for (const [id, timestamp] of [["first-hour", "2026-11-01T08:30:00Z"], ["second-hour", "2026-11-01T09:30:00Z"], ["before-midnight", "2026-11-01T18:14:59Z"], ["midnight", "2026-11-01T18:15:00Z"]]) {
      yield* store.record(record(id!, { completedAt: Date.parse(timestamp!) }))
    }
    return [yield* store.read(query), yield* store.read({ ...query, timeZone: "Asia/Kathmandu" })]
  }).pipe(Effect.provide(layer(root()))))
  vi.restoreAllMocks()
  expect(values[0]).toMatchObject({ dailyActivity: expect.arrayContaining([{ date: "2026-11-01", totalTokens: 480 }]) })
  expect(values[1]).toMatchObject({ dailyActivity: expect.arrayContaining([{ date: "2026-11-01", totalTokens: 360 }, { date: "2026-11-02", totalTokens: 120 }]) })
})
it("measures acceptance per model and drafter against plain decoding of the same model", async () => {
  const value = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("plain", { output: 40, generationMs: 1000 }))
    yield* store.record(record("drafted-1", { drafter: "hf:z/d@c/d.gguf", draftTokens: 100, acceptedDraftTokens: 60, output: 60, generationMs: 1000 }))
    yield* store.record(record("drafted-2", { drafter: "hf:z/d@c/d.gguf", draftTokens: 100, acceptedDraftTokens: 50, output: 60, generationMs: 1000 }))
    yield* store.record(record("slow", { model: ServingModelId.make("model-b"), drafter: "embedded", draftTokens: 10, acceptedDraftTokens: 1, output: 10, generationMs: 1000 }))
    yield* store.record(record("plain-b", { model: ServingModelId.make("model-b"), output: 20, generationMs: 1000 }))
    yield* store.record(record("no-drafting", { drafter: "embedded", draftTokens: null, acceptedDraftTokens: null }))
    return yield* store.acceptance
  }).pipe(Effect.provide(layer(root()))))
  expect(value).toEqual([
    { model: "model-a", drafter: "hf:z/d@c/d.gguf", acceptance: { acceptanceRate: 0.55, tokensPerSecond: 60, withoutDrafterTokensPerSecond: Option.some(40), slower: false } },
    { model: "model-b", drafter: "embedded", acceptance: { acceptanceRate: 0.1, tokensPerSecond: 10, withoutDrafterTokensPerSecond: Option.some(20), slower: true } },
  ])
})
it("migrates a version 0 ledger in place and keeps its history", async () => {
  const directory = root()
  const { Database } = await import("bun:sqlite")
  const legacy = new Database(join(directory, "serving-usage.sqlite"), { create: true, strict: true })
  legacy.exec(`CREATE TABLE usage (
    id TEXT PRIMARY KEY, model TEXT NOT NULL, completed_at REAL NOT NULL,
    input INTEGER, cached INTEGER, output INTEGER,
    generation_ms REAL, first_token_ms REAL, complete INTEGER NOT NULL
  ); CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id = 1), since REAL NOT NULL);`)
  legacy.query("INSERT INTO usage VALUES ('old', 'model-a', ?, 10, 0, 5, 100, 10, 1)").run(Date.now())
  legacy.query("INSERT INTO metadata VALUES (1, ?)").run(Date.now())
  legacy.close()
  const value = await Effect.runPromise(Effect.gen(function* () {
    const store = yield* ServingUsage
    yield* store.record(record("new", { drafter: "embedded", draftTokens: 4, acceptedDraftTokens: 2 }))
    return { usage: yield* store.read(query), acceptance: yield* store.acceptance }
  }).pipe(Effect.provide(layer(directory))))
  expect(value.usage).toMatchObject({ _tag: "Available", requests: 2 })
  expect(value.acceptance.map(({ drafter }) => drafter)).toEqual(["embedded"])
})
it("signals each newly recorded request", async () => {
  const signals = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const store = yield* ServingUsage
    const next = yield* store.changes.pipe(Stream.take(1), Stream.runCollect, Effect.fork)
    yield* Effect.sleep("10 millis")
    yield* store.record(record("signal"))
    return yield* Fiber.join(next)
  }).pipe(Effect.provide(layer(root())))))
  expect(signals.length).toBe(1)
})
