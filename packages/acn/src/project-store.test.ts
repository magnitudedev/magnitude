import { mkdtemp } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Either, Option } from "effect"
import {
  DirectoryPathSchema,
  ProjectIdSchema,
  ProjectPageCursorSchema,
  ProjectSchema,
  type Project,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type MagnitudeStorageShape } from "@magnitudedev/storage"
import { ProjectStore, ProjectStoreLive } from "./project-store"
import { makeTestStorageLayer } from "./session-test-support"

const project = (index: number, overrides?: Partial<Project>): Project => ProjectSchema.make({
  projectId: ProjectIdSchema.make(`project-${String(index).padStart(5, "0")}`),
  name: `Project ${index}`,
  cwd: DirectoryPathSchema.make(`/repos/project-${index}`),
  registrationState: "active",
  createdAt: index,
  updatedAt: index,
  ...overrides,
})

const run = <A, E>(
  body: (store: ProjectStore, storage: MagnitudeStorageShape) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(Effect.gen(function* () {
    const root = yield* Effect.promise(() =>
      mkdtemp(join(tmpdir(), "magnitude-project-store-")))
    const storageLayer = makeTestStorageLayer(root)
    return yield* Effect.gen(function* () {
      const store = yield* ProjectStore
      const storage = yield* MagnitudeStorage
      return yield* body(store, storage)
    }).pipe(
      Effect.provide(ProjectStoreLive),
      Effect.provide(storageLayer),
    )
  }))

describe("ProjectStore", () => {
  it("enforces cwd uniqueness across active and removed records", async () => {
    await run((store) => Effect.gen(function* () {
      const first = project(1)
      yield* store.insert(first)
      const duplicate = yield* Effect.either(store.insert(project(2, { cwd: first.cwd })))
      expect(Either.isLeft(duplicate)).toBe(true)
      if (Either.isLeft(duplicate)) {
        expect(duplicate.left._tag).toBe("ProjectCwdAlreadyRegistered")
      }

      yield* store.update(first.projectId, (current) => ({
        ...current,
        registrationState: "removed",
      }))
      // Removed records still hold their cwd registration.
      const stillTaken = yield* Effect.either(store.insert(project(3, { cwd: first.cwd })))
      expect(Either.isLeft(stillTaken)).toBe(true)
    }))
  })

  it("update rechecks cwd uniqueness at commit", async () => {
    await run((store) => Effect.gen(function* () {
      const first = project(1)
      const second = project(2)
      yield* store.insert(first)
      yield* store.insert(second)
      const conflicted = yield* Effect.either(
        store.update(second.projectId, (current) => ({ ...current, cwd: first.cwd })),
      )
      expect(Either.isLeft(conflicted)).toBe(true)
      if (Either.isLeft(conflicted)) {
        expect(conflicted.left._tag).toBe("ProjectCwdAlreadyRegistered")
      }
    }))
  })

  it("pages by recency with stable cursors and rejects invalid cursors", async () => {
    await run((store) => Effect.gen(function* () {
      for (let index = 1; index <= 5; index++) {
        yield* store.insert(project(index))
      }
      const first = yield* store.page({
        includeRemoved: false,
        cursor: Option.none(),
        limit: 2,
      })
      expect(first.items.map((item) => item.name)).toEqual(["Project 5", "Project 4"])
      expect(Option.isSome(first.nextCursor)).toBe(true)

      const second = yield* store.page({
        includeRemoved: false,
        cursor: first.nextCursor,
        limit: 2,
      })
      expect(second.items.map((item) => item.name)).toEqual(["Project 3", "Project 2"])

      const invalid = yield* Effect.either(store.page({
        includeRemoved: false,
        cursor: Option.some(ProjectPageCursorSchema.make("not-a-cursor")),
        limit: 2,
      }))
      expect(Either.isLeft(invalid)).toBe(true)
      if (Either.isLeft(invalid)) {
        expect(invalid.left._tag).toBe("InvalidProjectPageCursor")
      }
    }))
  })

  it("first page over ten thousand records stays bounded and never inspects the host", async () => {
    // Host isolation is structural: ProjectStoreLive requires only
    // MagnitudeStorage, so no filesystem or command service is even reachable.
    await run((store, storage) => Effect.gen(function* () {
      const projects = Array.from({ length: 10_000 }, (_, index) => project(index + 1))
      yield* storage.projects.update(() => ({ projects }))
      const page = yield* store.page({ includeRemoved: false, cursor: Option.none(), limit: 20 })
      expect(page.items).toHaveLength(20)
      expect(page.items[0]?.name).toBe("Project 10000")
      expect(Option.isSome(page.nextCursor)).toBe(true)
    }))
  })
})
