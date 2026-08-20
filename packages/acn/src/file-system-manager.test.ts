import { mkdir, mkdtemp, readFile, symlink, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, it } from "vitest"
import { Effect, Either } from "effect"
import { DirectoryPathSchema, RelativePathSchema } from "@magnitudedev/acn-protocol"
import { FileSystemManager, type OpenedDirectory } from "./file-system-manager"
import { testFileSystemManagerLayer } from "./session-test-support"

const path = RelativePathSchema.make

const withOpenRoot = <A, E>(
  body: (open: OpenedDirectory, root: string) => Effect.Effect<A, E>,
): Promise<A> =>
  Effect.runPromise(Effect.gen(function* () {
    const root = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-fsm-")))
    return yield* Effect.gen(function* () {
      const fileSystem = yield* FileSystemManager
      const open = yield* fileSystem.openDirectory(DirectoryPathSchema.make(root))
      return yield* body(open, root)
    }).pipe(Effect.provide(testFileSystemManagerLayer))
  }))

describe("FileSystemManager", () => {
  it("normalizes directory input lexically and rejects unusable input", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const fileSystem = yield* FileSystemManager
      expect(yield* fileSystem.normalizeDirectory("/a/b/../c/")).toBe("/a/c")
      const empty = yield* Effect.either(fileSystem.normalizeDirectory("   "))
      expect(Either.isLeft(empty)).toBe(true)
    }).pipe(Effect.provide(testFileSystemManagerLayer)))
  })

  it("opening a missing or non-directory path fails with its precise error", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const root = yield* Effect.promise(() => mkdtemp(join(tmpdir(), "magnitude-fsm-")))
      yield* Effect.promise(() => writeFile(join(root, "file.txt"), "x"))
      const fileSystem = yield* FileSystemManager
      const missing = yield* Effect.either(
        fileSystem.openDirectory(DirectoryPathSchema.make(join(root, "missing"))),
      )
      expect(Either.isLeft(missing)).toBe(true)
      if (Either.isLeft(missing)) expect(missing.left._tag).toBe("DirectoryNotFound")
      const file = yield* Effect.either(
        fileSystem.openDirectory(DirectoryPathSchema.make(join(root, "file.txt"))),
      )
      expect(Either.isLeft(file)).toBe(true)
      if (Either.isLeft(file)) expect(file.left._tag).toBe("PathNotDirectory")
    }).pipe(Effect.provide(testFileSystemManagerLayer)))
  })

  it("rejects symlinked targets and symlinked ancestors inside an opened root", async () => {
    await withOpenRoot((open, root) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await mkdir(join(root, "real"))
        await writeFile(join(root, "real", "file.txt"), "content")
        await symlink(join(root, "real"), join(root, "linked"))
        await symlink(join(root, "real", "file.txt"), join(root, "linked-file.txt"))
      })
      const linkedFile = yield* Effect.either(open.readFile(path("linked-file.txt")))
      expect(Either.isLeft(linkedFile)).toBe(true)
      if (Either.isLeft(linkedFile)) expect(linkedFile.left._tag).toBe("FileAccessDenied")

      const throughLinkedDirectory = yield* Effect.either(
        open.readFile(path("linked/file.txt")),
      )
      expect(Either.isLeft(throughLinkedDirectory)).toBe(true)
      if (Either.isLeft(throughLinkedDirectory)) {
        expect(throughLinkedDirectory.left._tag).toBe("FileAccessDenied")
      }

      const direct = yield* open.readFile(path("real/file.txt"))
      expect(new TextDecoder().decode(direct)).toBe("content")
    }))
  })

  it("listDirectory omits symlinks and reports kinds with sizes", async () => {
    await withOpenRoot((open, root) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await mkdir(join(root, "sub"))
        await writeFile(join(root, "a.txt"), "aaaa")
        await symlink(join(root, "a.txt"), join(root, "link.txt"))
      })
      const entries = yield* open.listDirectory(path(""))
      const names = entries.map((entry) => entry.name).sort()
      expect(names).toEqual(["a.txt", "sub"])
    }))
  })

  it("walkFiles is bounded and skips dot entries", async () => {
    await withOpenRoot((open, root) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await mkdir(join(root, ".hidden"))
        await writeFile(join(root, ".hidden", "secret.txt"), "x")
        await mkdir(join(root, "src"))
        for (let index = 0; index < 5; index++) {
          await writeFile(join(root, "src", `file-${index}.ts`), "x")
        }
      })
      const files = yield* open.walkFiles({ limit: 3 })
      expect(files).toHaveLength(3)
      expect(files.every((file) => !file.includes(".hidden"))).toBe(true)
    }))
  })

  it("writeFileAtomic aborts on guard failure and leaves the target untouched", async () => {
    await withOpenRoot((open, root) => Effect.gen(function* () {
      yield* Effect.promise(() => writeFile(join(root, "target.txt"), "before"))
      class GuardRejected extends Error {}
      const rejected = yield* Effect.either(open.writeFileAtomic(
        path("target.txt"),
        new TextEncoder().encode("after"),
        { guard: Effect.fail(new GuardRejected()) },
      ))
      expect(Either.isLeft(rejected)).toBe(true)
      expect(yield* Effect.promise(() => readFile(join(root, "target.txt"), "utf8"))).toBe("before")

      yield* open.writeFileAtomic(path("target.txt"), new TextEncoder().encode("after"))
      expect(yield* Effect.promise(() => readFile(join(root, "target.txt"), "utf8"))).toBe("after")
    }))
  })

  it("moveEntry never overwrites an existing destination", async () => {
    await withOpenRoot((open, root) => Effect.gen(function* () {
      yield* Effect.promise(async () => {
        await writeFile(join(root, "source.txt"), "source")
        await writeFile(join(root, "taken.txt"), "taken")
      })
      const conflict = yield* Effect.either(
        open.moveEntry(path("source.txt"), path("taken.txt")),
      )
      expect(Either.isLeft(conflict)).toBe(true)
      if (Either.isLeft(conflict)) expect(conflict.left._tag).toBe("FileAlreadyExists")

      yield* open.moveEntry(path("source.txt"), path("moved.txt"))
      expect(yield* Effect.promise(() => readFile(join(root, "moved.txt"), "utf8"))).toBe("source")
    }))
  })
})
