import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import {
  ProjectBusy,
  ProjectOperationFailed,
  type ListProjectsResult,
  type ProjectChange,
  type ProjectError,
  type ProjectGitState,
  type ProjectId,
  type ProjectRecord,
  type ProjectSummary,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Layer, Option, PubSub, Ref, Schedule, Stream } from "effect"
import { AgentRuntime } from "./agent-runtime"
import { ProjectRegistry } from "./project-registry"
import { SessionDrafts } from "./session-drafts"
import { SessionStore } from "./session-store"

export interface ProjectsApi {
  readonly list: (includeRemoved?: boolean) => Effect.Effect<ListProjectsResult, ProjectError>
  readonly create: (input: {
    readonly sourceDirectory: string
    readonly name: string
  }) => Effect.Effect<ProjectRecord, ProjectError>
  readonly edit: (input: {
    readonly projectId: ProjectId
    readonly sourceDirectory: string
    readonly name: string
  }) => Effect.Effect<ProjectRecord, ProjectError>
  readonly remove: (projectId: ProjectId) => Effect.Effect<ProjectRecord, ProjectError>
  readonly restore: (projectId: ProjectId) => Effect.Effect<ProjectRecord, ProjectError>
  readonly revealSource: (projectId: ProjectId) => Effect.Effect<void, ProjectError>
  readonly changes: Stream.Stream<ProjectChange, ProjectError>
}

export class Projects extends Context.Tag("Projects")<Projects, ProjectsApi>() {}

const operationFailed = (operation: string) => (cause: unknown) =>
  new ProjectOperationFailed({
    operation,
    reason: cause instanceof Error ? cause.message : String(cause),
  })

const revealKind = (): ListProjectsResult["revealKind"] => {
  if (process.platform === "darwin") return "finder"
  if (process.platform === "linux") return "folder"
  return "unsupported"
}

const gitOutput = (sourceDirectory: string, ...args: ReadonlyArray<string>) =>
  Command.make("git", "-C", sourceDirectory, ...args).pipe(
    Command.string,
    Effect.timeoutFail({
      duration: "3 seconds",
      onTimeout: () => new ProjectOperationFailed({
        operation: "inspect Git repository",
        reason: "Git inspection timed out",
      }),
    }),
    Effect.map((value) => value.trim()),
  )

const inspectGit = (
  sourceDirectory: string,
): Effect.Effect<ProjectGitState, never, CommandExecutor.CommandExecutor> =>
  gitOutput(
    sourceDirectory,
    "rev-parse",
    "--show-toplevel",
    "--abbrev-ref=strict",
    "HEAD",
  ).pipe(
    Effect.flatMap((output) => {
      const [rootDirectory, headName] = output.split(/\r?\n/)
      if (!rootDirectory || !headName) {
        return Effect.succeed<ProjectGitState>({
          _tag: "unavailable",
          message: "Git returned incomplete repository state",
        })
      }
      if (headName !== "HEAD") {
        return Effect.succeed<ProjectGitState>({
          _tag: "repository",
          rootDirectory,
          head: { _tag: "branch", name: headName },
        })
      }
      return gitOutput(sourceDirectory, "rev-parse", "--short", "HEAD").pipe(
        Effect.map((revision): ProjectGitState => ({
          _tag: "repository",
          rootDirectory,
          head: { _tag: "detached", revision },
        })),
      )
    }),
    Effect.catchAll(() => Effect.succeed<ProjectGitState>({ _tag: "not_repository" })),
  )

export const ProjectsLive: Layer.Layer<
  Projects,
  never,
  | AgentRuntime
  | ProjectRegistry
  | SessionDrafts
  | SessionStore
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
> = Layer.scoped(
  Projects,
  Effect.gen(function* () {
    const registry = yield* ProjectRegistry
    const store = yield* SessionStore
    const drafts = yield* SessionDrafts
    const runtime = yield* AgentRuntime
    const commandExecutor = yield* CommandExecutor.CommandExecutor
    const fs = yield* FileSystem.FileSystem
    const notifications = yield* PubSub.sliding<ProjectChange>(1)

    const publishChange = PubSub.publish(notifications, {}).pipe(Effect.asVoid)

    const inspectDirectory = (sourceDirectory: string) =>
      fs.stat(sourceDirectory).pipe(
        Effect.map((entry) => entry.type === "Directory"
          ? ({ _tag: "available" } as const)
          : ({ _tag: "inaccessible", message: "Source path is not a directory" } as const)),
        Effect.catchAll((error) => Effect.succeed(
          error._tag === "SystemError" && error.reason === "NotFound"
            ? ({ _tag: "missing" } as const)
            : ({ _tag: "inaccessible", message: String(error) } as const),
        )),
      )

    const inspectGitAvailability = Command.make("git", "--version").pipe(
      Command.string,
      Effect.timeoutFail({
        duration: "3 seconds",
        onTimeout: () => new ProjectOperationFailed({
          operation: "inspect Git availability",
          reason: "Git inspection timed out",
        }),
      }),
      Effect.match({ onFailure: () => false, onSuccess: () => true }),
    )

    const inspectProject = Effect.fn("acn.projects.inspect-project")(function* (
      project: ProjectRecord,
      gitAvailable: boolean,
    ) {
      const directoryState = yield* inspectDirectory(project.sourceDirectory)
      const gitState: ProjectGitState = directoryState._tag !== "available"
        ? { _tag: "unavailable", message: "Source directory is unavailable" }
        : !gitAvailable
          ? { _tag: "unavailable", message: "Git is unavailable" }
          : yield* inspectGit(project.sourceDirectory).pipe(
              Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
            )
      return { directoryState, gitState }
    })

    const list = Effect.fn("acn.projects.list")(function* (includeRemoved = false) {
      const records = yield* registry.list(includeRemoved).pipe(
        Effect.mapError(operationFailed("list projects")),
      )
      const sessions = yield* store.listAllProtocolMetas().pipe(
        Effect.mapError(operationFailed("list project sessions")),
      )
      const gitAvailable = records.length === 0
        ? false
        : yield* inspectGitAvailability.pipe(
            Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
          )
      const summaries = yield* Effect.forEach(records, (project) => Effect.gen(function* () {
        const projectSessions = sessions.filter(
          (session) => session.projectId === project.projectId,
        )
        const observation = yield* inspectProject(project, gitAvailable)
        return {
          project,
          ...observation,
          openSessionCount: projectSessions.filter((session) => session.sidebarOpen).length,
          totalSessionCount: projectSessions.length,
          recentActivityAt: projectSessions.reduce(
            (latest, session) => Math.max(latest, session.updatedAt),
            project.createdAt,
          ),
        } satisfies ProjectSummary
      }), { concurrency: 4 })
      summaries.sort((left, right) =>
        right.recentActivityAt - left.recentActivityAt ||
        left.project.name.localeCompare(right.project.name),
      )
      return { projects: summaries, revealKind: revealKind() }
    })

    const edit = Effect.fn("acn.projects.edit")(function* (input: {
      readonly projectId: ProjectId
      readonly sourceDirectory: string
      readonly name: string
    }) {
      return yield* registry.edit(input, () => Effect.gen(function* () {
        const projectSessionIds = new Set(
          (yield* store.listAllProtocolMetas().pipe(
            Effect.mapError(operationFailed("list project sessions before editing")),
          ))
            .filter((session) => session.projectId === input.projectId)
            .map((session) => session.sessionId),
        )
        const residents = (yield* runtime.residentSessions).filter((session) =>
          projectSessionIds.has(session.sessionId),
        )
        if (residents.some((session) => session.workStatus._tag === "Working")) {
          return yield* new ProjectBusy({ projectId: input.projectId })
        }
        if (!(yield* drafts.releaseProject(input.projectId).pipe(
          Effect.mapError(operationFailed("release project drafts")),
        ))) {
          return yield* new ProjectBusy({ projectId: input.projectId })
        }
        for (const resident of residents) yield* runtime.dispose(resident.sessionId)
      }))
    })

    const observationFingerprint = Effect.gen(function* () {
      const records = yield* registry.list()
      const gitAvailable = records.length === 0
        ? false
        : yield* inspectGitAvailability.pipe(
            Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
          )
      const observations = yield* Effect.forEach(records, (project) =>
        inspectProject(project, gitAvailable).pipe(
          Effect.map((observation) => ({
            projectId: project.projectId,
            sourceDirectory: project.sourceDirectory,
            ...observation,
          })),
        ), { concurrency: 4 })
      return observations.map((observation) => {
        const directory = observation.directoryState._tag === "inaccessible"
          ? `${observation.directoryState._tag}:${observation.directoryState.message}`
          : observation.directoryState._tag
        const git = observation.gitState._tag === "repository"
          ? observation.gitState.head._tag === "branch"
            ? `repository:${observation.gitState.rootDirectory}:branch:${observation.gitState.head.name}`
            : `repository:${observation.gitState.rootDirectory}:detached:${observation.gitState.head.revision}`
          : observation.gitState._tag === "unavailable"
            ? `unavailable:${observation.gitState.message}`
            : observation.gitState._tag
        return `${observation.projectId}\u0000${observation.sourceDirectory}\u0000${directory}\u0000${git}`
      }).join("\u0001")
    })

    const previousObservation = yield* Ref.make(Option.none<string>())
    yield* Effect.gen(function* () {
      const next = yield* observationFingerprint
      const previous = yield* Ref.getAndSet(previousObservation, Option.some(next))
      if (Option.isSome(previous) && previous.value !== next) yield* publishChange
    }).pipe(
      Effect.catchAllCause((cause) =>
        Effect.logWarning("Failed to observe project host state").pipe(
          Effect.annotateLogs({ cause: String(cause) }),
        ),
      ),
      Effect.repeat(Schedule.spaced("10 seconds")),
      Effect.forkScoped,
    )

    yield* Stream.mergeAll([
      registry.changes,
      runtime.changes,
      store.changes,
    ], { concurrency: "unbounded" }).pipe(
      Stream.runForEach(() => publishChange),
      Effect.forkScoped,
    )

    return Projects.of({
      list,
      create: (input) => registry.create(input),
      edit,
      remove: (projectId) => registry.remove(projectId),
      restore: (projectId) => registry.restore(projectId),
      revealSource: Effect.fn("acn.projects.reveal-source")(function* (projectId) {
        const project = yield* registry.get(projectId)
        const command = process.platform === "darwin"
          ? Command.make("open", "-R", project.sourceDirectory)
          : process.platform === "linux"
            ? Command.make("xdg-open", project.sourceDirectory)
            : null
        if (command === null) {
          return yield* new ProjectOperationFailed({
            operation: "reveal project source",
            reason: "This agent host does not support revealing folders",
          })
        }
        yield* Command.string(command).pipe(
          Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
          Effect.mapError(operationFailed("reveal project source")),
        )
      }),
      changes: Stream.fromPubSub(notifications),
    })
  }),
)
