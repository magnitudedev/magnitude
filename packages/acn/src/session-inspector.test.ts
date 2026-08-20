import { mkdtemp, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Either, Layer, Option, Ref } from "effect"
import {
  DirectoryPathSchema,
  SessionPageCursorSchema,
  type DirectoryPath,
  type SessionPageRequest,
} from "@magnitudedev/acn-protocol"
import {
  MagnitudeStorage,
  makeGlobalStoragePaths,
  type MagnitudeStorageShape,
  type StoredSessionMeta,
} from "@magnitudedev/storage"
import { SessionInspector, SessionInspectorLive } from "./session-inspector"
import { makeTestStorageLayer } from "./session-test-support"

const meta = (
  sessionId: string,
  cwd: DirectoryPath,
  overrides?: Partial<StoredSessionMeta>,
): StoredSessionMeta => ({
  sessionId,
  archived: false,
  pinnedAt: Option.none(),
  created: "2026-01-01T00:00:00.000Z",
  updated: "2026-01-01T00:00:00.000Z",
  chatName: `Chat ${sessionId}`,
  workingDirectory: cwd,
  visibility: "visible",
  initialVersion: "0.0.1",
  lastActiveVersion: "0.0.1",
  gitBranch: null,
  firstUserMessage: null,
  lastMessage: null,
  messageCount: 0,
  ...overrides,
})

const request = (overrides?: Partial<SessionPageRequest>): SessionPageRequest => ({
  cwd: Option.none(),
  archive: "active",
  pin: "all",
  query: Option.none(),
  cursor: Option.none(),
  limit: 50,
  ...overrides,
})

interface Setup {
  readonly inspector: SessionInspector
  readonly storage: MagnitudeStorageShape
  readonly root: string
  readonly metaReads: Ref.Ref<number>
}

const withInspector = <A, E>(body: (setup: Setup) => Effect.Effect<A, E>): Promise<A> =>
  Effect.runPromise(Effect.gen(function* () {
    const root = yield* Effect.promise(() =>
      mkdtemp(join(tmpdir(), "magnitude-session-inspector-")))
    const storage = yield* MagnitudeStorage.pipe(Effect.provide(makeTestStorageLayer(root)))
    const metaReads = yield* Ref.make(0)
    const counting: MagnitudeStorageShape = {
      ...storage,
      sessions: {
        ...storage.sessions,
        readMeta: (sessionId) => Ref.update(metaReads, (count) => count + 1).pipe(
          Effect.zipRight(storage.sessions.readMeta(sessionId)),
        ),
      },
    }
    return yield* Effect.gen(function* () {
      const inspector = yield* SessionInspector
      return yield* body({ inspector, storage, root, metaReads })
    }).pipe(Effect.provide(SessionInspectorLive.pipe(
      Layer.provide(Layer.succeed(MagnitudeStorage, counting)),
    )))
  }))

describe("SessionInspector", () => {
  it("get returns visible metadata, hides drafts, and reports unreadable targets", async () => {
    await withInspector((setup) => Effect.gen(function* () {
      const cwd = DirectoryPathSchema.make(setup.root)
      yield* setup.storage.sessions.writeMeta("visible-1", meta("visible-1", cwd))
      yield* setup.storage.sessions.writeMeta(
        "draft-1",
        meta("draft-1", cwd, { visibility: "draft" }),
      )

      const found = yield* setup.inspector.get("visible-1")
      expect(found.title).toBe("Chat visible-1")
      expect(found.cwd).toBe(cwd)

      const draft = yield* Effect.either(setup.inspector.get("draft-1"))
      expect(Either.isLeft(draft)).toBe(true)
      if (Either.isLeft(draft)) expect(draft.left._tag).toBe("SessionNotFound")

      // Corrupt one record on disk: get must report it explicitly.
      const paths = makeGlobalStoragePaths(join(setup.root, ".magnitude-test-data"))
      yield* Effect.promise(() =>
        writeFile(paths.sessionMetaFile("visible-1"), "not json"))
      const unreadable = yield* Effect.either(setup.inspector.get("visible-1"))
      expect(Either.isLeft(unreadable)).toBe(true)
      if (Either.isLeft(unreadable)) {
        expect(unreadable.left._tag).toBe("SessionMetadataUnreadable")
      }
    }))
  })

  it("filters, orders, and pages with fingerprinted cursors", async () => {
    await withInspector((setup) => Effect.gen(function* () {
      const cwd = DirectoryPathSchema.make(setup.root)
      // listSessionIds only surfaces sortable session ids.
      for (let index = 1; index <= 6; index++) {
        yield* setup.storage.sessions.writeMeta(`sess0${index}`, meta(`sess0${index}`, cwd, {
          updated: `2026-01-0${index}T00:00:00.000Z`,
          archived: index === 6,
        }))
      }
      const first = yield* setup.inspector.page(request({ limit: 2 }))
      expect(first.items.map((item) => item.sessionId)).toEqual(["sess05", "sess04"])
      expect(Option.isSome(first.nextCursor)).toBe(true)

      const second = yield* setup.inspector.page(request({ limit: 2, cursor: first.nextCursor }))
      expect(second.items.map((item) => item.sessionId)).toEqual(["sess03", "sess02"])

      // Reusing the cursor under different predicates is rejected outright.
      const mismatched = yield* Effect.either(setup.inspector.page(request({
        limit: 2,
        archive: "archived",
        cursor: first.nextCursor,
      })))
      expect(Either.isLeft(mismatched)).toBe(true)
      if (Either.isLeft(mismatched)) {
        expect(mismatched.left._tag).toBe("InvalidSessionPageCursor")
      }

      const garbage = yield* Effect.either(setup.inspector.page(request({
        cursor: Option.some(SessionPageCursorSchema.make("garbage")),
      })))
      expect(Either.isLeft(garbage)).toBe(true)

      const archived = yield* setup.inspector.page(request({ archive: "archived" }))
      expect(archived.items.map((item) => item.sessionId)).toEqual(["sess06"])

      const searched = yield* setup.inspector.page(request({
        query: Option.some("chat sess02"),
      }))
      expect(searched.items.map((item) => item.sessionId)).toEqual(["sess02"])
    }))
  })

  it("skips unreadable records in pages with a structured warning", async () => {
    await withInspector((setup) => Effect.gen(function* () {
      const cwd = DirectoryPathSchema.make(setup.root)
      yield* setup.storage.sessions.writeMeta("good01", meta("good01", cwd))
      yield* setup.storage.sessions.writeMeta("bad001", meta("bad001", cwd, {
        created: "not-a-timestamp",
      }))
      const page = yield* setup.inspector.page(request())
      expect(page.items.map((item) => item.sessionId)).toEqual(["good01"])
    }))
  })

  it("exact-cwd paging reads metadata proportional to the page, not the store", async () => {
    await withInspector((setup) => Effect.gen(function* () {
      const target = DirectoryPathSchema.make(join(setup.root, "target"))
      const other = DirectoryPathSchema.make(join(setup.root, "other"))
      // Recency order in the index: later writes land at the front.
      for (let index = 1; index <= 40; index++) {
        yield* setup.storage.sessions.writeMeta(`t-${index}`, meta(`t-${index}`, target, {
          updated: `2026-01-01T00:00:${String(index).padStart(2, "0")}.000Z`,
        }))
      }
      for (let index = 1; index <= 200; index++) {
        yield* setup.storage.sessions.writeMeta(`o-${index}`, meta(`o-${index}`, other))
      }
      yield* Ref.set(setup.metaReads, 0)
      const page = yield* setup.inspector.page(request({
        cwd: Option.some(target),
        limit: 5,
      }))
      expect(page.items.map((item) => item.sessionId)).toEqual(
        ["t-40", "t-39", "t-38", "t-37", "t-36"],
      )
      const reads = yield* Ref.get(setup.metaReads)
      // One index chunk of sixteen, never the 240-record store.
      expect(reads).toBeLessThanOrEqual(16)
    }))
  })

  it("aggregates recent directories with bounded pages", async () => {
    await withInspector((setup) => Effect.gen(function* () {
      const alpha = DirectoryPathSchema.make(join(setup.root, "alpha"))
      const beta = DirectoryPathSchema.make(join(setup.root, "beta"))
      yield* setup.storage.sessions.writeMeta("alpha1", meta("alpha1", alpha, {
        updated: "2026-01-01T00:00:00.000Z",
      }))
      yield* setup.storage.sessions.writeMeta("alpha2", meta("alpha2", alpha, {
        updated: "2026-01-03T00:00:00.000Z",
      }))
      yield* setup.storage.sessions.writeMeta("beta01", meta("beta01", beta, {
        updated: "2026-01-02T00:00:00.000Z",
      }))

      const first = yield* setup.inspector.recentDirectories({
        cursor: Option.none(),
        limit: 1,
      })
      expect(first.items).toEqual([
        { cwd: alpha, lastActiveAt: Date.parse("2026-01-03T00:00:00.000Z"), sessionCount: 2 },
      ])
      expect(Option.isSome(first.nextCursor)).toBe(true)

      const second = yield* setup.inspector.recentDirectories({
        cursor: first.nextCursor,
        limit: 1,
      })
      expect(second.items.map((item) => item.cwd)).toEqual([beta])
    }))
  })
})
