import { mkdtemp, mkdir } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Either, Layer } from "effect"
import { ProjectManager, ProjectManagerLive } from "./project-manager"
import { ProjectStoreLive } from "./project-store"
import { makeTestStorageLayer, testFileSystemManagerLayer } from "./session-test-support"

const run = <A, E>(
  body: (manager: ProjectManager, root: string) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(Effect.gen(function* () {
    const root = yield* Effect.promise(() =>
      mkdtemp(join(tmpdir(), "magnitude-project-manager-")))
    const layer = ProjectManagerLive.pipe(
      Layer.provide(Layer.mergeAll(
        ProjectStoreLive.pipe(Layer.provide(makeTestStorageLayer(root))),
        testFileSystemManagerLayer,
      )),
    )
    return yield* Effect.gen(function* () {
      const manager = yield* ProjectManager
      return yield* body(manager, root)
    }).pipe(Effect.provide(layer))
  }))

describe("ProjectManager", () => {
  it("creates, is idempotent for an active cwd, and validates the host directory", async () => {
    await run((manager, root) => Effect.gen(function* () {
      const created = yield* manager.create({ cwd: root, name: "  Alpha  " })
      expect(created.name).toBe("Alpha")
      expect(created.cwd).toBe(root)

      const again = yield* manager.create({ cwd: root, name: "Different" })
      expect(again.projectId).toBe(created.projectId)
      expect(again.name).toBe("Alpha")

      const missing = yield* Effect.either(
        manager.create({ cwd: join(root, "does-not-exist"), name: "Nope" }),
      )
      expect(Either.isLeft(missing)).toBe(true)
      if (Either.isLeft(missing)) expect(missing.left._tag).toBe("DirectoryNotFound")

      const unnamed = yield* Effect.either(manager.create({ cwd: root, name: "   " }))
      expect(Either.isLeft(unnamed)).toBe(true)
      if (Either.isLeft(unnamed)) expect(unnamed.left._tag).toBe("InvalidProjectName")
    }))
  })

  it("remove flips registration state only and create restores the same identity", async () => {
    await run((manager, root) => Effect.gen(function* () {
      const created = yield* manager.create({ cwd: root, name: "Alpha" })
      const removed = yield* manager.remove(created.projectId)
      expect(removed.registrationState).toBe("removed")

      const restored = yield* manager.create({ cwd: root, name: "Beta" })
      expect(restored.projectId).toBe(created.projectId)
      expect(restored.registrationState).toBe("active")
      expect(restored.name).toBe("Beta")
    }))
  })

  it("edit rebinds name and cwd without touching anything else", async () => {
    await run((manager, root) => Effect.gen(function* () {
      const other = join(root, "other")
      yield* Effect.promise(() => mkdir(other))
      const created = yield* manager.create({ cwd: root, name: "Alpha" })
      const edited = yield* manager.edit({
        projectId: created.projectId,
        cwd: other,
        name: "Alpha Two",
      })
      expect(edited.projectId).toBe(created.projectId)
      expect(edited.cwd).toBe(other)
      expect(edited.name).toBe("Alpha Two")
      expect(edited.createdAt).toBe(created.createdAt)
    }))
  })

  it("restore is idempotent and reports the restored record", async () => {
    await run((manager, root) => Effect.gen(function* () {
      const created = yield* manager.create({ cwd: root, name: "Alpha" })
      yield* manager.remove(created.projectId)
      const restored = yield* manager.restore(created.projectId)
      expect(restored.registrationState).toBe("active")
      const again = yield* manager.restore(created.projectId)
      expect(again.registrationState).toBe("active")
    }))
  })
})
