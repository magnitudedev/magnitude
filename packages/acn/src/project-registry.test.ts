import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { Deferred, Effect, Fiber, Layer, Option } from "effect"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import {
  GlobalStorage,
  MagnitudeStorage,
  ProjectStorage,
  StorageLive,
  Version,
  makeGlobalStoragePaths,
  makeProjectStoragePaths,
} from "@magnitudedev/storage"
import { ProjectRegistry, ProjectRegistryLive } from "./project-registry"

const VERSION = "0.0.1"

const makeLayer = (root: string) => {
  const base = Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => VERSION })),
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root),
    })),
    Layer.succeed(ProjectStorage, ProjectStorage.of({
      cwd: root,
      root: join(root, "project-storage"),
      paths: makeProjectStoragePaths(root),
    })),
  )
  const storage = StorageLive.pipe(Layer.provide(base))
  return Layer.provideMerge(ProjectRegistryLive, Layer.merge(storage, base))
}

describe("ProjectRegistry", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-project-registry-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("migrates a legacy session to one durable project before normal reads", async () => {
    const paths = makeGlobalStoragePaths(root)
    const source = join(root, "source")
    await mkdir(source, { recursive: true })
    await mkdir(paths.sessionDir("session-1"), { recursive: true })
    await Bun.write(paths.sessionMetaFile("session-1"), JSON.stringify({
      sessionId: "session-1",
      created: "2026-01-01T00:00:00.000Z",
      updated: "2026-01-01T00:00:00.000Z",
      chatName: "Legacy session",
      workingDirectory: source,
      visibility: "visible",
    }))
    await mkdir(paths.sessionDir("damaged-session"), { recursive: true })
    await Bun.write(paths.sessionMetaFile("damaged-session"), "{broken")

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const registry = yield* ProjectRegistry
        const storage = yield* MagnitudeStorage
        return {
          projects: yield* registry.list(),
          meta: yield* storage.sessions.readMeta("session-1"),
        }
      }).pipe(Effect.provide(makeLayer(root))),
    )

    expect(result.projects).toHaveLength(1)
    expect(result.projects[0]?.name).toBe("source")
    expect(result.meta?.projectId).toBe(result.projects[0]?.projectId)
  })

  it("fails visibly when durable session identity references a missing project", async () => {
    const paths = makeGlobalStoragePaths(root)
    const source = join(root, "source")
    await mkdir(source, { recursive: true })
    await mkdir(paths.sessionDir("session-1"), { recursive: true })
    await Bun.write(paths.sessionMetaFile("session-1"), JSON.stringify({
      sessionId: "session-1",
      projectId: "missing-project",
      created: "2026-01-01T00:00:00.000Z",
      updated: "2026-01-01T00:00:00.000Z",
      chatName: "Orphaned session",
      workingDirectory: source,
      visibility: "visible",
    }))

    const result = await Effect.runPromise(
      Effect.scoped(ProjectRegistry.pipe(
        Effect.provide(makeLayer(root)),
        Effect.either,
      )),
    )

    expect(result._tag).toBe("Left")
    if (result._tag === "Left" && result.left._tag === "ProjectOperationFailed") {
      expect(result.left.reason).toContain("references missing project")
    }
  })

  it("rejects duplicate durable project identities instead of choosing an authority", async () => {
    const paths = makeGlobalStoragePaths(root)
    const sourceA = join(root, "source-a")
    const sourceB = join(root, "source-b")
    await mkdir(sourceA, { recursive: true })
    await mkdir(sourceB, { recursive: true })
    await Bun.write(paths.projectsFile, JSON.stringify({
      projects: [
        {
          projectId: "duplicate",
          name: "A",
          sourceDirectory: sourceA,
          registrationState: "active",
          createdAt: 1,
          updatedAt: 1,
        },
        {
          projectId: "duplicate",
          name: "B",
          sourceDirectory: sourceB,
          registrationState: "active",
          createdAt: 2,
          updatedAt: 2,
        },
      ],
    }))

    const result = await Effect.runPromise(
      Effect.scoped(ProjectRegistry.pipe(
        Effect.provide(makeLayer(root)),
        Effect.either,
      )),
    )

    expect(result._tag).toBe("Left")
    if (result._tag === "Left" && result.left._tag === "ProjectOperationFailed") {
      expect(result.left.reason).toContain("Duplicate project identity")
    }
  })

  it("enforces canonical source uniqueness and restores removed identity", async () => {
    const source = join(root, "source")
    await mkdir(source, { recursive: true })

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const registry = yield* ProjectRegistry
        const created = yield* registry.create({ sourceDirectory: source, name: "First" })
        const duplicate = yield* registry.create({
          sourceDirectory: join(source, "."),
          name: "Duplicate",
        })
        const removed = yield* registry.remove(created.projectId)
        const removedAgain = yield* registry.remove(created.projectId)
        const restoredByCwd = yield* registry.ensureForSourceDirectory(source)
        yield* registry.remove(created.projectId)
        const restored = yield* registry.create({ sourceDirectory: source, name: "Renamed" })
        const replayedEdit = yield* registry.edit(
          {
            projectId: restored.projectId,
            sourceDirectory: source,
            name: "Renamed",
          },
          () => Effect.die("no-op edit must not prepare a source rebind"),
        )
        return { created, duplicate, removed, removedAgain, restoredByCwd, restored, replayedEdit }
      }).pipe(Effect.provide(makeLayer(root))),
    )

    expect(result.duplicate).toEqual(result.created)
    expect(result.removedAgain).toEqual(result.removed)
    expect(result.restoredByCwd.registrationState).toBe("removed")
    expect(result.restored.projectId).toBe(result.created.projectId)
    expect(result.restored.name).toBe("Renamed")
    expect(result.restored.registrationState).toBe("active")
    expect(result.replayedEdit).toEqual(result.restored)
  })

  it("serializes source resolution while a rebind prepares its commit", async () => {
    const original = join(root, "original")
    const replacement = join(root, "replacement")
    await mkdir(original, { recursive: true })
    await mkdir(replacement, { recursive: true })

    await Effect.runPromise(
      Effect.gen(function* () {
        const registry = yield* ProjectRegistry
        const created = yield* registry.create({ sourceDirectory: original, name: "Project" })
        const entered = yield* Deferred.make<void>()
        const release = yield* Deferred.make<void>()
        const edit = yield* registry.edit(
          {
            projectId: created.projectId,
            sourceDirectory: replacement,
            name: "Project",
          },
          () => Deferred.succeed(entered, undefined).pipe(
            Effect.zipRight(Deferred.await(release)),
          ),
        ).pipe(Effect.fork)
        yield* Deferred.await(entered)
        const resolution = yield* registry.resolveSourceDirectory(created.projectId).pipe(
          Effect.fork,
        )
        yield* Effect.yieldNow()
        expect(Option.isNone(yield* Fiber.poll(resolution))).toBe(true)
        yield* Deferred.succeed(release, undefined)
        const edited = yield* Fiber.join(edit)
        expect(yield* Fiber.join(resolution)).toBe(edited.sourceDirectory)
      }).pipe(Effect.provide(makeLayer(root))),
    )
  })
})
