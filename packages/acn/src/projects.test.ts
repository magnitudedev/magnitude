import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { BunCommandExecutor, BunFileSystem } from "@effect/platform-bun"
import { Effect, Layer, Option, Stream } from "effect"
import {
  ProjectIdSchema,
  type ProjectRecord,
} from "@magnitudedev/acn-protocol"
import { AgentRuntime, type AgentRuntimeApi } from "./agent-runtime"
import {
  ProjectRegistry,
  type ProjectRegistryApi,
} from "./project-registry"
import { Projects, ProjectsLive } from "./projects"
import { SessionDrafts, type SessionDraftsApi } from "./session-drafts"
import { SessionStore, type SessionStoreApi } from "./session-store"

const runGit = async (cwd: string, ...args: ReadonlyArray<string>): Promise<void> => {
  const process = Bun.spawn(["git", "-C", cwd, ...args], {
    stdout: "pipe",
    stderr: "pipe",
  })
  const exitCode = await process.exited
  if (exitCode !== 0) throw new Error(await new Response(process.stderr).text())
}

const makeProject = (sourceDirectory: string): ProjectRecord => ({
  projectId: ProjectIdSchema.make("project-test"),
  name: "Project test",
  sourceDirectory,
  registrationState: "active",
  createdAt: 1,
  updatedAt: 2,
})

const makeLayer = (
  project: ProjectRecord,
  options: { readonly releaseProject?: boolean } = {},
) => {
  const registry: ProjectRegistryApi = {
    list: () => Effect.succeed([project]),
    get: () => Effect.succeed(project),
    resolveSourceDirectory: () => Effect.succeed(project.sourceDirectory),
    create: () => Effect.die("unused"),
    edit: (input, prepareSourceRebind) => prepareSourceRebind(
      project,
      input.sourceDirectory,
    ).pipe(Effect.as({ ...project, ...input })),
    remove: () => Effect.die("unused"),
    restore: () => Effect.die("unused"),
    ensureForSourceDirectory: () => Effect.die("unused"),
    changes: Stream.never,
  }
  const store: SessionStoreApi = {
    createId: Effect.die("unused"),
    readMeta: () => Effect.die("unused"),
    readProtocolMeta: () => Effect.die("unused"),
    promoteDraft: () => Effect.die("unused"),
    listDraftSessionIds: () => Effect.die("unused"),
    listProtocolMetas: () => Effect.die("unused"),
    listAllProtocolMetas: () => Effect.succeed([]),
    listSessionCwds: () => Effect.die("unused"),
    deleteSessionFiles: () => Effect.die("unused"),
    validateCwd: Effect.succeed,
    getScratchpadPath: () => Effect.die("unused"),
    getExecutionContext: () => Effect.die("unused"),
    ensureProjectForCwd: () => Effect.die("unused"),
    resolveProjectSource: () => Effect.die("unused"),
    setSidebarOpen: () => Effect.die("unused"),
    changes: Stream.never,
  }
  const runtime: AgentRuntimeApi = {
    withSession: () => Effect.die("unused"),
    withSessionRequest: () => Effect.die("unused"),
    tryWithResident: () => Effect.succeed(Option.none()),
    tryWithBusyResident: () => Effect.succeed(Option.none()),
    residentSessions: Effect.succeed([]),
    dispose: () => Effect.void,
    deleteSession: (_sessionId, remove) => remove,
    registerRetirementObserver: () => Effect.succeed(Effect.void),
    changes: Stream.never,
  }
  const drafts: SessionDraftsApi = {
    preload: () => Effect.die("unused"),
    release: () => Effect.die("unused"),
    claim: () => Effect.die("unused"),
    promote: () => Effect.die("unused"),
    releaseClaim: () => Effect.die("unused"),
    releaseProject: () => Effect.succeed(options.releaseProject ?? true),
  }
  return ProjectsLive.pipe(Layer.provide(Layer.mergeAll(
    Layer.succeed(ProjectRegistry, registry),
    Layer.succeed(SessionStore, store),
    Layer.succeed(AgentRuntime, runtime),
    Layer.succeed(SessionDrafts, drafts),
    BunFileSystem.layer,
    BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
  )))
}

describe("Projects", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-projects-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("derives the current Git branch and directory status on the agent host", async () => {
    await runGit(root, "init", "-b", "project-test")
    await runGit(root, "config", "user.email", "test@magnitude.dev")
    await runGit(root, "config", "user.name", "Magnitude Test")
    await writeFile(join(root, "README.md"), "test\n")
    await runGit(root, "add", "README.md")
    await runGit(root, "commit", "-m", "Initial")

    const result = await Effect.runPromise(
      Effect.scoped(Effect.gen(function* () {
        const projects = yield* Projects
        return yield* projects.list()
      }).pipe(Effect.provide(makeLayer(makeProject(root))))),
    )

    expect(result.projects).toHaveLength(1)
    expect(result.projects[0]?.directoryState).toEqual({ _tag: "available" })
    expect(result.projects[0]?.gitState).toMatchObject({
      _tag: "repository",
      head: { _tag: "branch", name: "project-test" },
    })
    expect(result.revealKind).toBe(process.platform === "darwin" ? "finder" : "folder")
  })

  it("keeps a missing source as project history instead of failing the list", async () => {
    const missing = join(root, "missing")
    await mkdir(root, { recursive: true })
    const result = await Effect.runPromise(
      Effect.scoped(Effect.gen(function* () {
        const projects = yield* Projects
        return yield* projects.list()
      }).pipe(Effect.provide(makeLayer(makeProject(missing))))),
    )

    expect(result.projects[0]?.directoryState).toEqual({ _tag: "missing" })
    expect(result.projects[0]?.gitState._tag).toBe("unavailable")
  })

  it("rejects a source rebind while a draft claim is in flight", async () => {
    const replacement = join(root, "replacement")
    await mkdir(replacement, { recursive: true })
    const project = makeProject(root)
    const result = await Effect.runPromise(
      Effect.scoped(Effect.gen(function* () {
        const projects = yield* Projects
        return yield* projects.edit({
          projectId: project.projectId,
          name: project.name,
          sourceDirectory: replacement,
        }).pipe(Effect.either)
      }).pipe(Effect.provide(makeLayer(project, { releaseProject: false })))),
    )

    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("ProjectBusy")
  })
})
