import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, test, beforeEach, afterEach } from "vitest"
import { Effect, Layer } from "effect"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import {
  GlobalStorage,
  MagnitudeStorage,
  ProjectStorage,
  StorageLive,
  Version,
  makeGlobalStoragePaths,
  makeProjectStoragePaths,
  type StoredSessionMeta,
} from "@magnitudedev/storage"
import { SessionStore, SessionStoreLive } from "./session-store"

const VERSION = "0.0.1"

function makeTestLayer(root: string) {
  const base = Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => VERSION })),
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root),
    })),
    Layer.succeed(ProjectStorage, ProjectStorage.of({
      cwd: "/repo",
      root: join(root, "project"),
      paths: makeProjectStoragePaths(root),
    })),
  )
  const storageLayer = StorageLive.pipe(Layer.provide(base))
  return Layer.provideMerge(SessionStoreLive, storageLayer)
}

const run = <A, E>(eff: Effect.Effect<A, E, SessionStore | MagnitudeStorage>, root: string) =>
  Effect.runPromise(eff.pipe(Effect.provide(makeTestLayer(root))))

const meta = (
  sessionId: string,
  updated: string,
  workingDirectory = "/repo",
  visibility: StoredSessionMeta["visibility"] = "visible",
): StoredSessionMeta => ({
  sessionId,
  chatName: sessionId,
  workingDirectory,
  visibility,
  gitBranch: null,
  created: "2026-01-01T00:00:00.000Z",
  updated,
  initialVersion: VERSION,
  lastActiveVersion: VERSION,
  firstUserMessage: null,
  lastMessage: null,
  messageCount: 0,
})

describe("SessionStore", () => {
  let tmpDir: string

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "magnitude-acn-session-store-"))
    await mkdir(tmpDir, { recursive: true })
  })

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true })
  })

  test("lists sessions by updated time with cursor pagination", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        yield* storage.sessions.writeMeta("mqa00000", meta("mqa00000", "2026-01-02T00:00:00.000Z"))
        yield* storage.sessions.writeMeta("mqa00001", meta("mqa00001", "2026-01-04T00:00:00.000Z"))
        yield* storage.sessions.writeMeta("mqa00002", meta("mqa00002", "2026-01-03T00:00:00.000Z"))

        const store = yield* SessionStore
        const firstPage = yield* store.listProtocolMetas({ limit: 2 })
        const secondPage = firstPage.nextCursor._tag === "Some"
          ? yield* store.listProtocolMetas({ cursor: firstPage.nextCursor.value, limit: 2 })
          : yield* Effect.die("missing cursor")

        return { firstPage, secondPage }
      }),
      tmpDir
    ).then(({ firstPage, secondPage }) => {
      expect(firstPage.items.map((session) => session.sessionId)).toEqual(["mqa00001", "mqa00002"])
      expect(firstPage.hasMore).toBe(true)
      expect(firstPage.nextCursor._tag).toBe("Some")
      expect(secondPage.items.map((session) => session.sessionId)).toEqual(["mqa00000"])
      expect(secondPage.hasMore).toBe(false)
    })
  })

  test("filters sessions by cwd and query on the server", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        yield* storage.sessions.writeMeta("mqa00000", meta("mqa00000", "2026-01-02T00:00:00.000Z", "/repo"))
        yield* storage.sessions.writeMeta("mqa00001", meta("mqa00001", "2026-01-04T00:00:00.000Z", "/repo"))
        yield* storage.sessions.writeMeta("mqa00002", meta("mqa00002", "2026-01-03T00:00:00.000Z", "/other"))

        const store = yield* SessionStore
        return yield* store.listProtocolMetas({ cwd: "/repo", query: "0000", limit: 10 })
      }),
      tmpDir
    ).then((page) => {
      expect(page.items.map((session) => session.sessionId)).toEqual(["mqa00001", "mqa00000"])
      expect(page.hasMore).toBe(false)
    })
  })

  test("lists cwd summaries sorted by recent activity", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        yield* storage.sessions.writeMeta("mqa00000", meta("mqa00000", "2026-01-02T00:00:00.000Z", "/repo"))
        yield* storage.sessions.writeMeta("mqa00001", meta("mqa00001", "2026-01-04T00:00:00.000Z", "/other"))
        yield* storage.sessions.writeMeta("mqa00002", meta("mqa00002", "2026-01-03T00:00:00.000Z", "/repo"))

        const store = yield* SessionStore
        return yield* store.listSessionCwds()
      }),
      tmpDir
    ).then((summaries) => {
      expect(summaries.map((summary) => summary.cwd)).toEqual(["/other", "/repo"])
      expect(summaries.map((summary) => summary.sessionCount)).toEqual([1, 2])
    })
  })

  test("hides draft sessions from protocol lists while exposing them for cleanup", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage
        yield* storage.sessions.writeMeta("mqa00000", meta("mqa00000", "2026-01-02T00:00:00.000Z", "/repo", "visible"))
        yield* storage.sessions.writeMeta("mqa00001", meta("mqa00001", "2026-01-04T00:00:00.000Z", "/repo", "draft"))
        yield* storage.sessions.writeMeta("mqa00002", meta("mqa00002", "2026-01-03T00:00:00.000Z", "/other", "draft"))

        const store = yield* SessionStore
        const sessions = yield* store.listProtocolMetas({ limit: 10 })
        const cwdSummaries = yield* store.listSessionCwds()
        const visibleMeta = yield* store.readProtocolMeta("mqa00000")
        const draftMeta = yield* store.readProtocolMeta("mqa00001")
        const draftIds = yield* store.listDraftSessionIds()
        return { sessions, cwdSummaries, visibleMeta, draftMeta, draftIds }
      }),
      tmpDir
    ).then(({ sessions, cwdSummaries, visibleMeta, draftMeta, draftIds }) => {
      expect(sessions.items.map((session) => session.sessionId)).toEqual(["mqa00000"])
      expect(cwdSummaries.map((summary) => summary.cwd)).toEqual(["/repo"])
      expect(cwdSummaries.map((summary) => summary.sessionCount)).toEqual([1])
      expect(visibleMeta?.sessionId).toBe("mqa00000")
      // readProtocolMeta now returns metadata for any session (including drafts).
      // Visibility filtering belongs in the session list, not in readProtocolMeta.
      expect(draftMeta?.sessionId).toBe("mqa00001")
      expect([...draftIds].sort()).toEqual(["mqa00001", "mqa00002"])
    })
  })
})
