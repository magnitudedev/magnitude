import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Either, Layer } from "effect"
import {
  DirectoryPathSchema,
  ProjectIdSchema,
  ProjectSchema,
  RelativePathSchema,
  type FileContentHash,
  type ProjectId,
} from "@magnitudedev/acn-protocol"
import { ProjectFileManager, ProjectFileManagerLive } from "./project-file-manager"
import { ProjectStore, ProjectStoreLive } from "./project-store"
import { makeTestStorageLayer, testFileSystemManagerLayer } from "./session-test-support"

const path = RelativePathSchema.make

interface Setup {
  readonly manager: ProjectFileManager
  readonly projectId: ProjectId
  readonly secondProjectId: ProjectId
  readonly root: string
  readonly secondRoot: string
}

const withManager = <A, E>(body: (setup: Setup) => Effect.Effect<A, E>): Promise<A> =>
  Effect.runPromise(Effect.gen(function* () {
    const base = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-pfm-")))
    const root = join(base, "alpha")
    const secondRoot = join(base, "beta")
    yield* Effect.promise(async () => {
      await mkdir(root)
      await mkdir(secondRoot)
    })
    const storeLayer = ProjectStoreLive.pipe(Layer.provide(makeTestStorageLayer(base)))
    const layer = Layer.provideMerge(
      ProjectFileManagerLive,
      Layer.mergeAll(storeLayer, testFileSystemManagerLayer),
    )
    return yield* Effect.gen(function* () {
      const store = yield* ProjectStore
      const projectId = ProjectIdSchema.make("project-alpha")
      const secondProjectId = ProjectIdSchema.make("project-beta")
      yield* store.insert(ProjectSchema.make({
        projectId,
        name: "Alpha",
        cwd: DirectoryPathSchema.make(root),
        registrationState: "active",
        createdAt: 1,
        updatedAt: 1,
      }))
      yield* store.insert(ProjectSchema.make({
        projectId: secondProjectId,
        name: "Beta",
        cwd: DirectoryPathSchema.make(secondRoot),
        registrationState: "active",
        createdAt: 1,
        updatedAt: 1,
      }))
      const manager = yield* ProjectFileManager
      return yield* body({ manager, projectId, secondProjectId, root, secondRoot })
    }).pipe(Effect.provide(layer))
  }))

const textHash = (setup: Setup, filePath: string): Effect.Effect<FileContentHash> =>
  setup.manager.readFile(setup.projectId, path(filePath)).pipe(
    Effect.flatMap((snapshot) => snapshot._tag === "text"
      ? Effect.succeed(snapshot.contentHash)
      : Effect.die(new Error(`expected a text snapshot, got ${snapshot._tag}`))),
    Effect.orDie,
  )

describe("ProjectFileManager", () => {
  it("lists directories sorted with .git filtered and reads text snapshots", async () => {
    await withManager((setup) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await mkdir(join(setup.root, "src"))
        await mkdir(join(setup.root, ".git"))
        await writeFile(join(setup.root, "readme.md"), "# hi\n")
        await writeFile(join(setup.root, "src", "index.ts"), "export {}\n")
      })
      const listing = yield* setup.manager.listDirectory(setup.projectId, path(""))
      expect(listing.entries.map((entry) => entry.name)).toEqual(["src", "readme.md"])

      const snapshot = yield* setup.manager.readFile(setup.projectId, path("readme.md"))
      expect(snapshot._tag).toBe("text")
      if (snapshot._tag === "text") {
        expect(snapshot.content).toBe("# hi\n")
        expect(snapshot.newline).toBe("lf")
      }
    }))
  })

  it("writes atomically with stale-hash protection", async () => {
    await withManager((setup) => Effect.gen(function* () {
      yield* Effect.promise(() => writeFile(join(setup.root, "notes.md"), "one"))
      const contentHash = yield* textHash(setup, "notes.md")

      const written = yield* setup.manager.writeFile({
        projectId: setup.projectId,
        path: path("notes.md"),
        content: "two",
        expectedContentHash: contentHash,
      })
      expect(written.content).toBe("two")
      expect(yield* Effect.promise(() =>
        readFile(join(setup.root, "notes.md"), "utf8"))).toBe("two")

      // The original hash is now stale: nothing may be written.
      const stale = yield* Effect.either(setup.manager.writeFile({
        projectId: setup.projectId,
        path: path("notes.md"),
        content: "three",
        expectedContentHash: contentHash,
      }))
      expect(Either.isLeft(stale)).toBe(true)
      if (Either.isLeft(stale)) {
        expect(stale.left._tag).toBe("ProjectFileChanged")
        if (stale.left._tag === "ProjectFileChanged") {
          expect(stale.left.current.content).toBe("two")
        }
      }
      expect(yield* Effect.promise(() =>
        readFile(join(setup.root, "notes.md"), "utf8"))).toBe("two")
    }))
  })

  it("deletes only when the hash still matches", async () => {
    await withManager((setup) => Effect.gen(function* () {
      yield* Effect.promise(() => writeFile(join(setup.root, "notes.md"), "one"))
      const contentHash = yield* textHash(setup, "notes.md")
      yield* Effect.promise(() => writeFile(join(setup.root, "notes.md"), "changed"))

      const stale = yield* Effect.either(setup.manager.deleteFile({
        projectId: setup.projectId,
        path: path("notes.md"),
        expectedContentHash: contentHash,
      }))
      expect(Either.isLeft(stale)).toBe(true)
      if (Either.isLeft(stale)) expect(stale.left._tag).toBe("ProjectFileChanged")

      const fresh = yield* textHash(setup, "notes.md")
      yield* setup.manager.deleteFile({
        projectId: setup.projectId,
        path: path("notes.md"),
        expectedContentHash: fresh,
      })
      const gone = yield* Effect.either(setup.manager.readFile(setup.projectId, path("notes.md")))
      expect(Either.isLeft(gone)).toBe(true)
      if (Either.isLeft(gone)) expect(gone.left._tag).toBe("ProjectFileNotFound")
    }))
  })

  it("moves without overwriting and guards self-moves", async () => {
    await withManager((setup) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await mkdir(join(setup.root, "docs"))
        await writeFile(join(setup.root, "notes.md"), "n")
        await writeFile(join(setup.root, "docs", "taken.md"), "t")
      })
      const moved = yield* setup.manager.moveEntry({
        projectId: setup.projectId,
        sourcePath: path("notes.md"),
        destinationDirectory: path("docs"),
      })
      expect(moved.destinationPath).toBe("docs/notes.md")

      yield* Effect.promise(() => writeFile(join(setup.root, "taken.md"), "other"))
      const conflict = yield* Effect.either(setup.manager.moveEntry({
        projectId: setup.projectId,
        sourcePath: path("taken.md"),
        destinationDirectory: path("docs"),
      }))
      expect(Either.isLeft(conflict)).toBe(true)
      if (Either.isLeft(conflict)) expect(conflict.left._tag).toBe("ProjectFileAlreadyExists")

      const selfMove = yield* Effect.either(setup.manager.moveEntry({
        projectId: setup.projectId,
        sourcePath: path("docs"),
        destinationDirectory: path("docs"),
      }))
      expect(Either.isLeft(selfMove)).toBe(true)
    }))
  })

  it("coordinates writes per project cwd so unrelated projects never serialize", async () => {
    await withManager((setup) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await writeFile(join(setup.root, "a.md"), "a")
        await writeFile(join(setup.secondRoot, "b.md"), "b")
      })
      const hashA = yield* textHash(setup, "a.md")
      const snapshotB = yield* setup.manager.readFile(setup.secondProjectId, path("b.md"))
      if (snapshotB._tag !== "text") throw new Error("expected text")

      // Both writes run concurrently; each commits against its own root.
      yield* Effect.all([
        setup.manager.writeFile({
          projectId: setup.projectId,
          path: path("a.md"),
          content: "a2",
          expectedContentHash: hashA,
        }),
        setup.manager.writeFile({
          projectId: setup.secondProjectId,
          path: path("b.md"),
          content: "b2",
          expectedContentHash: snapshotB.contentHash,
        }),
      ], { concurrency: "unbounded" })
      expect(yield* Effect.promise(() => readFile(join(setup.root, "a.md"), "utf8"))).toBe("a2")
      expect(yield* Effect.promise(() =>
        readFile(join(setup.secondRoot, "b.md"), "utf8"))).toBe("b2")
    }))
  })
})
