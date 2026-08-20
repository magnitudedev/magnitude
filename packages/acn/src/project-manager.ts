import {
  InvalidProjectName,
  ProjectCwdAlreadyRegistered,
  ProjectIdSchema,
  ProjectNotFound,
  ProjectSchema,
  ProjectStoreUnavailable,
  RevealFailed,
  RevealUnsupported,
  type DirectoryAccessDenied,
  type DirectoryNotFound,
  type FileSystemUnavailable,
  type InvalidDirectoryPath,
  type PathNotDirectory,
  type Project,
  type ProjectId,
} from "@magnitudedev/acn-protocol"
import { createId } from "@magnitudedev/generate-id"
import { Clock, Context, Effect, Layer, Option } from "effect"
import { FileSystemManager } from "./file-system-manager"
import { ProjectStore } from "./project-store"

export type CreateProjectError =
  | InvalidProjectName
  | InvalidDirectoryPath
  | DirectoryNotFound
  | DirectoryAccessDenied
  | PathNotDirectory
  | FileSystemUnavailable
  | ProjectCwdAlreadyRegistered
  | ProjectStoreUnavailable

export interface ProjectManager {
  readonly create: (request: {
    readonly cwd: string
    readonly name: string
  }) => Effect.Effect<Project, CreateProjectError>
  readonly edit: (request: {
    readonly projectId: ProjectId
    readonly cwd: string
    readonly name: string
  }) => Effect.Effect<Project, ProjectNotFound | CreateProjectError>
  readonly remove: (
    projectId: ProjectId,
  ) => Effect.Effect<Project, ProjectNotFound | ProjectStoreUnavailable>
  readonly restore: (
    projectId: ProjectId,
  ) => Effect.Effect<Project, ProjectNotFound | ProjectCwdAlreadyRegistered | ProjectStoreUnavailable>
  readonly reveal: (projectId: ProjectId) => Effect.Effect<
    void,
    ProjectNotFound | ProjectStoreUnavailable | RevealUnsupported | RevealFailed
  >
}

export const ProjectManager = Context.GenericTag<ProjectManager>("acn/ProjectManager")

const normalizedName = (name: string): Effect.Effect<string, InvalidProjectName> => {
  const value = name.trim()
  return value.length > 0 ? Effect.succeed(value) : Effect.fail(new InvalidProjectName({ name }))
}

export const ProjectManagerLive: Layer.Layer<
  ProjectManager,
  never,
  ProjectStore | FileSystemManager
> = Layer.effect(
  ProjectManager,
  Effect.gen(function* () {
    const store = yield* ProjectStore
    const fileSystem = yield* FileSystemManager

    return ProjectManager.of({
      create: Effect.fn("acn.project-manager.create")(function* (request) {
        const name = yield* normalizedName(request.name)
        const cwd = yield* fileSystem.normalizeDirectory(request.cwd)
        yield* fileSystem.openDirectory(cwd)
        const existing = yield* store.findByCwd(cwd, true)
        if (Option.isSome(existing)) {
          if (existing.value.registrationState === "active") return existing.value
          const now = yield* Clock.currentTimeMillis
          return yield* store.update(existing.value.projectId, (project) => ({
            ...project,
            name,
            registrationState: "active",
            updatedAt: now,
          })).pipe(
            // The record was just read and records are never physically removed;
            // its disappearance would be a broken store invariant, not a domain state.
            Effect.catchTag("ProjectNotFound", (error) => Effect.die(error)),
          )
        }
        const now = yield* Clock.currentTimeMillis
        return yield* store.insert(ProjectSchema.make({
          projectId: ProjectIdSchema.make(createId()),
          name,
          cwd,
          registrationState: "active",
          createdAt: now,
          updatedAt: now,
        }))
      }),
      edit: Effect.fn("acn.project-manager.edit")(function* (request) {
        const name = yield* normalizedName(request.name)
        const cwd = yield* fileSystem.normalizeDirectory(request.cwd)
        yield* fileSystem.openDirectory(cwd)
        const now = yield* Clock.currentTimeMillis
        return yield* store.update(request.projectId, (project) =>
          project.name === name && project.cwd === cwd
            ? project
            : { ...project, name, cwd, updatedAt: now })
      }),
      remove: Effect.fn("acn.project-manager.remove")(function* (projectId) {
        const now = yield* Clock.currentTimeMillis
        return yield* store.update(projectId, (project) =>
          project.registrationState === "removed"
            ? project
            : { ...project, registrationState: "removed", updatedAt: now }).pipe(
          // The transition never changes cwd, so a cwd conflict here would be a
          // broken uniqueness invariant, not a domain state.
          Effect.catchTag("ProjectCwdAlreadyRegistered", (error) => Effect.die(error)),
        )
      }),
      restore: Effect.fn("acn.project-manager.restore")(function* (projectId) {
        const project = yield* store.get(projectId)
        const duplicate = yield* store.findByCwd(project.cwd)
        if (Option.isSome(duplicate) && duplicate.value.projectId !== projectId) {
          return yield* new ProjectCwdAlreadyRegistered({
            projectId: duplicate.value.projectId,
            cwd: project.cwd,
          })
        }
        const now = yield* Clock.currentTimeMillis
        return yield* store.update(projectId, (current) =>
          current.registrationState === "active"
            ? current
            : { ...current, registrationState: "active", updatedAt: now })
      }),
      reveal: Effect.fn("acn.project-manager.reveal")(function* (projectId) {
        const project = yield* store.get(projectId)
        yield* fileSystem.revealDirectory(project.cwd)
      }),
    })
  }),
)
