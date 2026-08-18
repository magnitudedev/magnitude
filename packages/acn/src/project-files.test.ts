import { afterEach, describe, expect, it } from "vitest"
import * as BunFileSystem from "@effect/platform-bun/BunFileSystem"
import * as BunPath from "@effect/platform-bun/BunPath"
import { chmod, mkdtemp, mkdir, readFile, rename, rm, stat, symlink, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Cause, Effect, Fiber, Layer, Option, Stream } from "effect"
import { ProjectIdSchema, ProjectRelativePathSchema } from "@magnitudedev/acn-protocol"
import { ProjectFiles, ProjectFilesLive } from "./project-files"
import { ProjectRegistry } from "./project-registry"

const roots: string[] = []
afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))))

const fixture = async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-project-files-"))
  roots.push(root)
  await mkdir(join(root, "src"))
  await mkdir(join(root, ".git"))
  await writeFile(join(root, "src", "index.ts"), "export const answer = 41\n")
  await writeFile(join(root, "README.md"), "# Fixture\n\nHello.\n")
  return root
}

const projectId = ProjectIdSchema.make("project-files-test")
const layerFor = (root: string) => ProjectFilesLive.pipe(Layer.provideMerge(Layer.mergeAll(
  BunFileSystem.layer,
  BunPath.layer,
  Layer.succeed(ProjectRegistry, ProjectRegistry.of({
    list: () => Effect.succeed([]),
    get: () => Effect.die("unused"),
    resolveSourceDirectory: () => Effect.succeed(root),
    create: () => Effect.die("unused"),
    edit: () => Effect.die("unused"),
    remove: () => Effect.die("unused"),
    restore: () => Effect.die("unused"),
    ensureForSourceDirectory: () => Effect.die("unused"),
    changes: Stream.empty,
  })),
)))

const run = <A, E>(root: string, effect: Effect.Effect<A, E, ProjectFiles>) =>
  Effect.runPromise(effect.pipe(Effect.provide(layerFor(root))))

const observeExternalChange = (
  root: string,
  change: () => Promise<unknown>,
) => run(root, Effect.gen(function* () {
  const files = yield* ProjectFiles
  const event = yield* files.watchChanges(projectId).pipe(
    Stream.runHead,
    Effect.timeoutFail({
      duration: "5 seconds",
      onTimeout: () => new Error("project-files watch did not observe the external change"),
    }),
    Effect.fork,
  )
  yield* Effect.sleep("100 millis")
  yield* Effect.tryPromise(change)
  return yield* Fiber.join(event)
}))

describe("ProjectFiles", () => {
  it("lists sorted project entries without exposing .git", async () => {
    const root = await fixture()
    const listing = await run(root, Effect.flatMap(ProjectFiles, (files) => files.listDirectory(projectId, ProjectRelativePathSchema.make(""))))
    expect(listing.entries.map(({ name, kind }) => [name, kind])).toEqual([
      ["src", "directory"],
      ["README.md", "file"],
    ])
  })

  it("lists empty, deep, and wide directories one level at a time", async () => {
    const root = await fixture()
    await mkdir(join(root, "empty"))
    await mkdir(join(root, "deep", "one", "two"), { recursive: true })
    await writeFile(join(root, "deep", "one", "two", "leaf.ts"), "export {}\n")
    await mkdir(join(root, "wide"))
    await Promise.all(Array.from({ length: 150 }, (_, index) =>
      writeFile(join(root, "wide", `file-${String(index).padStart(3, "0")}.txt`), `${index}\n`)))
    const [empty, deep, wide] = await Promise.all([
      run(root, Effect.flatMap(ProjectFiles, (files) => files.listDirectory(projectId, ProjectRelativePathSchema.make("empty")))),
      run(root, Effect.flatMap(ProjectFiles, (files) => files.listDirectory(projectId, ProjectRelativePathSchema.make("deep")))),
      run(root, Effect.flatMap(ProjectFiles, (files) => files.listDirectory(projectId, ProjectRelativePathSchema.make("wide")))),
    ])
    expect(empty.entries).toEqual([])
    expect(deep.entries.map(({ name, kind }) => [name, kind])).toEqual([["one", "directory"]])
    expect(wide.entries).toHaveLength(150)
    expect(wide.entries[0]?.name).toBe("file-000.txt")
    expect(wide.entries.at(-1)?.name).toBe("file-149.txt")
  })

  it("observes external creates, edits, moves, and removals recursively", async () => {
    const root = await fixture()
    const created = join(root, "src", "external.ts")
    const moved = join(root, "external.ts")
    const observations = [
      await observeExternalChange(root, () => writeFile(created, "export const external = 1\n")),
      await observeExternalChange(root, () => writeFile(created, "export const external = 2\n")),
      await observeExternalChange(root, () => rename(created, moved)),
      await observeExternalChange(root, () => rm(moved)),
      await observeExternalChange(root, () => rename(
        join(root, "README.md"),
        join(root, "src", "README.md"),
      )),
    ]
    expect(observations.map(Option.getOrUndefined)).toEqual(
      Array.from({ length: observations.length }, () => ({ projectId })),
    )
  })

  it("reads UTF-8 code and Markdown snapshots", async () => {
    const root = await fixture()
    for (const path of ["src/index.ts", "README.md"] as const) {
      const snapshot = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, ProjectRelativePathSchema.make(path))))
      expect(snapshot._tag).toBe("text")
      if (snapshot._tag === "text") {
        expect(snapshot.path).toBe(path)
        expect(snapshot.revision).toMatch(/^[a-f0-9]{64}$/)
      }
    }
  })

  it("classifies previewable images, binary files, and newline styles", async () => {
    const root = await fixture()
    await writeFile(join(root, "pixel.png"), Uint8Array.from([0x89, 0x50, 0x4e, 0x47]))
    await writeFile(join(root, "binary.dat"), Uint8Array.from([0x01, 0x00, 0x02]))
    await writeFile(join(root, "windows.txt"), "first\r\nsecond\r\n")
    const paths = ["pixel.png", "binary.dat", "windows.txt"] as const
    const [image, binary, windows] = await Promise.all(paths.map((path) => run(root,
      Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, ProjectRelativePathSchema.make(path))),
    )))
    expect(image).toMatchObject({ _tag: "image", mediaType: "image/png", data: "iVBORw==" })
    expect(binary).toMatchObject({ _tag: "unsupported", reason: "binary" })
    expect(windows).toMatchObject({ _tag: "text", newline: "crlf" })
  })

  it("rejects oversized text without reading it into an editable snapshot", async () => {
    const root = await fixture()
    await writeFile(join(root, "oversized.txt"), "x".repeat(5 * 1024 * 1024 + 1))
    const snapshot = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(
      projectId,
      ProjectRelativePathSchema.make("oversized.txt"),
    )))
    expect(snapshot).toMatchObject({ _tag: "unsupported", reason: "too_large", size: 5 * 1024 * 1024 + 1 })
  })

  it("persists edits to the actual project file", async () => {
    const root = await fixture()
    const path = ProjectRelativePathSchema.make("src/index.ts")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    const saved = await run(root, Effect.flatMap(ProjectFiles, (files) => files.writeFile({
      projectId,
      path,
      content: "export const answer = 42\n",
      expectedRevision: initial.revision,
    })))
    expect(saved.content).toContain("42")
    expect(await readFile(join(root, "src", "index.ts"), "utf8")).toBe("export const answer = 42\n")
  })

  it("preserves file permissions across an atomic save", async () => {
    const root = await fixture()
    const absolute = join(root, "src", "index.ts")
    await chmod(absolute, 0o744)
    const path = ProjectRelativePathSchema.make("src/index.ts")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    await run(root, Effect.flatMap(ProjectFiles, (files) => files.writeFile({
      projectId,
      path,
      content: "export const answer = 42\n",
      expectedRevision: initial.revision,
    })))
    expect((await stat(absolute)).mode & 0o777).toBe(0o744)
  })

  it("serializes concurrent writes so only one stale revision can commit", async () => {
    const root = await fixture()
    const path = ProjectRelativePathSchema.make("README.md")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    const exits = await run(root, Effect.flatMap(ProjectFiles, (files) => Effect.all(
      ["first", "second"].map((content) => Effect.exit(files.writeFile({
        projectId,
        path,
        content,
        expectedRevision: initial.revision,
      }))),
      { concurrency: "unbounded" },
    )))
    expect(exits.filter((exit) => exit._tag === "Success")).toHaveLength(1)
    expect(exits.filter((exit) => exit._tag === "Failure")).toHaveLength(1)
    expect(["first", "second"]).toContain(await readFile(join(root, "README.md"), "utf8"))
  })

  it("returns the current snapshot instead of overwriting an external edit", async () => {
    const root = await fixture()
    const path = ProjectRelativePathSchema.make("README.md")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    await writeFile(join(root, "README.md"), "# Changed elsewhere\n")
    const exit = await Effect.runPromiseExit(
      Effect.flatMap(ProjectFiles, (files) => files.writeFile({ projectId, path, content: "mine", expectedRevision: initial.revision })).pipe(Effect.provide(layerFor(root))),
    )
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const failure = Cause.failureOption(exit.cause)
      expect(Option.isSome(failure) ? failure.value._tag : null).toBe("ProjectFileConflict")
    }
    expect(await readFile(join(root, "README.md"), "utf8")).toBe("# Changed elsewhere\n")
  })

  it("deletes a file only when its expected revision is current", async () => {
    const root = await fixture()
    const path = ProjectRelativePathSchema.make("README.md")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    await run(root, Effect.flatMap(ProjectFiles, (files) => files.deleteFile({
      projectId,
      path,
      expectedRevision: initial.revision,
    })))
    await expect(readFile(join(root, "README.md"), "utf8")).rejects.toMatchObject({ code: "ENOENT" })
  })

  it("deletes a previewable image using its content revision", async () => {
    const root = await fixture()
    const absolute = join(root, "pixel.png")
    await writeFile(absolute, Uint8Array.from([0x89, 0x50, 0x4e, 0x47]))
    const path = ProjectRelativePathSchema.make("pixel.png")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "image") throw new Error("expected image")
    await run(root, Effect.flatMap(ProjectFiles, (files) => files.deleteFile({
      projectId,
      path,
      expectedRevision: initial.revision,
    })))
    await expect(readFile(absolute)).rejects.toMatchObject({ code: "ENOENT" })
  })

  it("does not delete a file that changed after it was opened", async () => {
    const root = await fixture()
    const path = ProjectRelativePathSchema.make("README.md")
    const initial = await run(root, Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, path)))
    if (initial._tag !== "text") throw new Error("expected text")
    await writeFile(join(root, "README.md"), "# Changed before removal\n")
    const exit = await Effect.runPromiseExit(
      Effect.flatMap(ProjectFiles, (files) => files.deleteFile({ projectId, path, expectedRevision: initial.revision })).pipe(Effect.provide(layerFor(root))),
    )
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const failure = Cause.failureOption(exit.cause)
      expect(Option.isSome(failure) ? failure.value._tag : null).toBe("ProjectFileConflict")
    }
    expect(await readFile(join(root, "README.md"), "utf8")).toBe("# Changed before removal\n")
  })

  it("moves files into folders and back to the project root", async () => {
    const root = await fixture()
    const sourcePath = ProjectRelativePathSchema.make("README.md")
    const intoSource = await run(root, Effect.flatMap(ProjectFiles, (files) => files.moveEntry({
      projectId,
      sourcePath,
      destinationDirectory: ProjectRelativePathSchema.make("src"),
    })))
    expect(intoSource).toEqual({
      sourcePath: "README.md",
      destinationPath: "src/README.md",
      kind: "file",
    })
    expect(await readFile(join(root, "src", "README.md"), "utf8")).toBe("# Fixture\n\nHello.\n")
    await expect(readFile(join(root, "README.md"), "utf8")).rejects.toMatchObject({ code: "ENOENT" })

    const intoRoot = await run(root, Effect.flatMap(ProjectFiles, (files) => files.moveEntry({
      projectId,
      sourcePath: ProjectRelativePathSchema.make("src/README.md"),
      destinationDirectory: ProjectRelativePathSchema.make(""),
    })))
    expect(intoRoot.destinationPath).toBe("README.md")
    expect(await readFile(join(root, "README.md"), "utf8")).toBe("# Fixture\n\nHello.\n")
  })

  it("moves directories with their complete contents", async () => {
    const root = await fixture()
    await mkdir(join(root, "packages"))
    const moved = await run(root, Effect.flatMap(ProjectFiles, (files) => files.moveEntry({
      projectId,
      sourcePath: ProjectRelativePathSchema.make("src"),
      destinationDirectory: ProjectRelativePathSchema.make("packages"),
    })))
    expect(moved).toEqual({
      sourcePath: "src",
      destinationPath: "packages/src",
      kind: "directory",
    })
    expect(await readFile(join(root, "packages", "src", "index.ts"), "utf8")).toContain("answer")
    await expect(stat(join(root, "src"))).rejects.toMatchObject({ code: "ENOENT" })
  })

  it("rejects root, same-parent, descendant, and non-directory destinations", async () => {
    const root = await fixture()
    await mkdir(join(root, "src", "nested"))
    const attempts = [
      {
        sourcePath: ProjectRelativePathSchema.make(""),
        destinationDirectory: ProjectRelativePathSchema.make("src"),
        tag: "InvalidProjectFilePath",
      },
      {
        sourcePath: ProjectRelativePathSchema.make("README.md"),
        destinationDirectory: ProjectRelativePathSchema.make(""),
        tag: "ProjectFileAccessDenied",
      },
      {
        sourcePath: ProjectRelativePathSchema.make("src"),
        destinationDirectory: ProjectRelativePathSchema.make("src/nested"),
        tag: "ProjectFileAccessDenied",
      },
      {
        sourcePath: ProjectRelativePathSchema.make("README.md"),
        destinationDirectory: ProjectRelativePathSchema.make("src/index.ts"),
        tag: "ProjectFileAccessDenied",
      },
    ] as const
    for (const attempt of attempts) {
      const exit = await Effect.runPromiseExit(
        Effect.flatMap(ProjectFiles, (files) => files.moveEntry({ projectId, ...attempt })).pipe(
          Effect.provide(layerFor(root)),
        ),
      )
      expect(exit._tag).toBe("Failure")
      if (exit._tag === "Failure") {
        const failure = Cause.failureOption(exit.cause)
        expect(Option.isSome(failure) ? failure.value._tag : null).toBe(attempt.tag)
      }
    }
    expect(await readFile(join(root, "README.md"), "utf8")).toContain("Fixture")
    expect(await readFile(join(root, "src", "index.ts"), "utf8")).toContain("answer")
  })

  it("reports missing move sources and destinations without changing the tree", async () => {
    const root = await fixture()
    for (const input of [
      {
        sourcePath: ProjectRelativePathSchema.make("missing.txt"),
        destinationDirectory: ProjectRelativePathSchema.make("src"),
      },
      {
        sourcePath: ProjectRelativePathSchema.make("README.md"),
        destinationDirectory: ProjectRelativePathSchema.make("missing"),
      },
    ]) {
      const exit = await Effect.runPromiseExit(
        Effect.flatMap(ProjectFiles, (files) => files.moveEntry({ projectId, ...input })).pipe(
          Effect.provide(layerFor(root)),
        ),
      )
      expect(exit._tag).toBe("Failure")
      if (exit._tag === "Failure") {
        const failure = Cause.failureOption(exit.cause)
        expect(Option.isSome(failure) ? failure.value._tag : null).toBe("ProjectFileNotFound")
      }
    }
    expect(await readFile(join(root, "README.md"), "utf8")).toContain("Fixture")
  })

  it("never overwrites an existing file or directory at the destination", async () => {
    const root = await fixture()
    await mkdir(join(root, "target"))
    await writeFile(join(root, "target", "README.md"), "existing\n")
    await mkdir(join(root, "target", "src"))
    for (const sourcePath of ["README.md", "src"] as const) {
      const exit = await Effect.runPromiseExit(
        Effect.flatMap(ProjectFiles, (files) => files.moveEntry({
          projectId,
          sourcePath: ProjectRelativePathSchema.make(sourcePath),
          destinationDirectory: ProjectRelativePathSchema.make("target"),
        })).pipe(Effect.provide(layerFor(root))),
      )
      expect(exit._tag).toBe("Failure")
      if (exit._tag === "Failure") {
        const failure = Cause.failureOption(exit.cause)
        expect(Option.isSome(failure) ? failure.value._tag : null).toBe("ProjectFileAlreadyExists")
      }
    }
    expect(await readFile(join(root, "target", "README.md"), "utf8")).toBe("existing\n")
    expect(await readFile(join(root, "README.md"), "utf8")).toContain("Fixture")
    expect(await readFile(join(root, "src", "index.ts"), "utf8")).toContain("answer")
  })

  it("serializes competing moves so only one can claim a destination", async () => {
    const root = await fixture()
    await mkdir(join(root, "one"))
    await mkdir(join(root, "two"))
    await mkdir(join(root, "target"))
    await writeFile(join(root, "one", "same.txt"), "one")
    await writeFile(join(root, "two", "same.txt"), "two")
    const exits = await run(root, Effect.flatMap(ProjectFiles, (files) => Effect.all(
      ["one/same.txt", "two/same.txt"].map((sourcePath) => Effect.exit(files.moveEntry({
        projectId,
        sourcePath: ProjectRelativePathSchema.make(sourcePath),
        destinationDirectory: ProjectRelativePathSchema.make("target"),
      }))),
      { concurrency: "unbounded" },
    )))
    expect(exits.filter((exit) => exit._tag === "Success")).toHaveLength(1)
    expect(exits.filter((exit) => exit._tag === "Failure")).toHaveLength(1)
    expect(["one", "two"]).toContain(await readFile(join(root, "target", "same.txt"), "utf8"))
  })

  it("rejects symbolic-link sources and destinations", async () => {
    const root = await fixture()
    await mkdir(join(root, "target"))
    await symlink(join(root, "README.md"), join(root, "linked.md"))
    await symlink(join(root, "target"), join(root, "linked-target"))
    for (const input of [
      {
        sourcePath: ProjectRelativePathSchema.make("linked.md"),
        destinationDirectory: ProjectRelativePathSchema.make("target"),
      },
      {
        sourcePath: ProjectRelativePathSchema.make("README.md"),
        destinationDirectory: ProjectRelativePathSchema.make("linked-target"),
      },
    ]) {
      const exit = await Effect.runPromiseExit(
        Effect.flatMap(ProjectFiles, (files) => files.moveEntry({ projectId, ...input })).pipe(
          Effect.provide(layerFor(root)),
        ),
      )
      expect(exit._tag).toBe("Failure")
      if (exit._tag === "Failure") expect(String(exit.cause)).toContain("ProjectFileAccessDenied")
    }
  })

  it("rejects symlinks even when their target remains inside the project", async () => {
    const root = await fixture()
    await symlink(join(root, "README.md"), join(root, "linked.md"))
    const exit = await Effect.runPromiseExit(
      Effect.flatMap(ProjectFiles, (files) => files.readFile(projectId, ProjectRelativePathSchema.make("linked.md"))).pipe(Effect.provide(layerFor(root))),
    )
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") expect(String(exit.cause)).toContain("ProjectFileAccessDenied")
  })
})
