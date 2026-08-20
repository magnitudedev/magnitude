import { describe, expect, it } from "vitest"
import { Effect, Layer, Ref, Stream } from "effect"
import {
  DirectoryPathSchema,
  ProjectIdSchema,
  ProjectSchema,
  type DirectoryInspection,
} from "@magnitudedev/acn-protocol"
import { FileSystemManager } from "./file-system-manager"
import { GitInspector } from "./git-inspector"
import { ProjectInspector, ProjectInspectorLive } from "./project-inspector"
import { ProjectStore } from "./project-store"

const project = ProjectSchema.make({
  projectId: ProjectIdSchema.make("project-1"),
  name: "Alpha",
  cwd: DirectoryPathSchema.make("/repos/alpha"),
  registrationState: "active",
  createdAt: 1,
  updatedAt: 1,
})

const run = (directory: DirectoryInspection) =>
  Effect.runPromise(Effect.gen(function* () {
    const directoryInspections = yield* Ref.make(0)
    const gitInspections = yield* Ref.make(0)
    const store: ProjectStore = {
      get: () => Effect.succeed(project),
      findByCwd: () => Effect.die("unused"),
      page: () => Effect.die("unused"),
      insert: () => Effect.die("unused"),
      update: () => Effect.die("unused"),
      changes: Stream.never,
    }
    const fileSystem: FileSystemManager = {
      normalizeDirectory: () => Effect.die("unused"),
      resolveHostPath: () => Effect.die("unused"),
      inspectDirectory: () => Ref.update(directoryInspections, (count) => count + 1).pipe(
        Effect.as(directory),
      ),
      inspectPath: () => Effect.die("unused"),
      openDirectory: () => Effect.die("unused"),
      readHostFile: () => Effect.die("unused"),
      watchHostFile: () => Stream.die("unused"),
      listHostSubdirectories: () => Effect.die("unused"),
      revealDirectory: () => Effect.die("unused"),
    }
    const git: GitInspector = {
      inspect: () => Ref.update(gitInspections, (count) => count + 1).pipe(
        Effect.as({
          _tag: "git_repository" as const,
          rootDirectory: project.cwd,
          head: { _tag: "branch" as const, name: "main" },
        }),
      ),
      recentFiles: () => Effect.die("unused"),
    }
    const inspection = yield* Effect.gen(function* () {
      const inspector = yield* ProjectInspector
      return yield* inspector.inspect(project.projectId)
    }).pipe(Effect.provide(ProjectInspectorLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(ProjectStore, store),
      Layer.succeed(FileSystemManager, fileSystem),
      Layer.succeed(GitInspector, git),
    )))))
    return {
      inspection,
      directoryInspections: yield* Ref.get(directoryInspections),
      gitInspections: yield* Ref.get(gitInspections),
    }
  }))

describe("ProjectInspector", () => {
  it("inspects git exactly once for an available directory", async () => {
    const result = await run({ _tag: "available" })
    expect(result.directoryInspections).toBe(1)
    expect(result.gitInspections).toBe(1)
    expect(result.inspection.directory).toEqual({ _tag: "available" })
    expect(result.inspection.git._tag).toBe("git_repository")
  })

  it("runs zero git commands when the directory is missing", async () => {
    const result = await run({ _tag: "missing" })
    expect(result.gitInspections).toBe(0)
    expect(result.inspection.directory).toEqual({ _tag: "missing" })
    expect(result.inspection.git).toEqual({ _tag: "git_inspection_failed" })
  })
})
