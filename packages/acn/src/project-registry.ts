import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Effect, Layer, Option, Stream } from "effect"
import { createId } from "@magnitudedev/generate-id"
import {
  InvalidProjectName,
  InvalidProjectSource,
  ProjectIdSchema,
  ProjectNotFound,
  ProjectOperationFailed,
  ProjectSourceAlreadyRegistered,
  type ProjectError,
  type ProjectId,
  type ProjectRecord,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage } from "@magnitudedev/storage"

export interface ProjectRegistryApi {
  readonly list: (includeRemoved?: boolean) => Effect.Effect<ReadonlyArray<ProjectRecord>>
  readonly get: (projectId: ProjectId) => Effect.Effect<ProjectRecord, ProjectError>
  readonly resolveSourceDirectory: (projectId: ProjectId) => Effect.Effect<string, ProjectError>
  readonly create: (input: {
    readonly sourceDirectory: string
    readonly name: string
  }) => Effect.Effect<ProjectRecord, ProjectError>
  readonly edit: (
    input: {
      readonly projectId: ProjectId
      readonly sourceDirectory: string
      readonly name: string
    },
    prepareSourceRebind: (
      current: ProjectRecord,
      nextSourceDirectory: string,
    ) => Effect.Effect<void, ProjectError>,
  ) => Effect.Effect<ProjectRecord, ProjectError>
  readonly remove: (projectId: ProjectId) => Effect.Effect<ProjectRecord, ProjectError>
  readonly restore: (projectId: ProjectId) => Effect.Effect<ProjectRecord, ProjectError>
  readonly ensureForSourceDirectory: (
    sourceDirectory: string,
    options?: { readonly allowUnavailable?: boolean },
  ) => Effect.Effect<ProjectRecord, ProjectError>
  readonly changes: Stream.Stream<void>
}

export class ProjectRegistry extends Context.Tag("ProjectRegistry")<
  ProjectRegistry,
  ProjectRegistryApi
>() {}

const operationFailed = (operation: string) => (cause: unknown) =>
  new ProjectOperationFailed({ operation, reason: String(cause) })

const normalizeName = (name: string): Effect.Effect<string, InvalidProjectName> => {
  const normalized = name.trim()
  return normalized.length > 0
    ? Effect.succeed(normalized)
    : Effect.fail(new InvalidProjectName({ name }))
}

const canonicalSource = (
  fs: FileSystem.FileSystem,
  path: Path.Path,
  sourceDirectory: string,
  allowUnavailable: boolean,
): Effect.Effect<string, InvalidProjectSource> => {
  const absolute = path.resolve(sourceDirectory)
  return Effect.gen(function* () {
    const info = yield* fs.stat(absolute).pipe(
      Effect.mapError((cause) => new InvalidProjectSource({
        path: absolute,
        reason: cause.message,
      })),
    )
    if (info.type !== "Directory") {
      return yield* new InvalidProjectSource({
        path: absolute,
        reason: "path is not a directory",
      })
    }
    return yield* fs.realPath(absolute).pipe(
      Effect.mapError((cause) => new InvalidProjectSource({
        path: absolute,
        reason: cause.message,
      })),
    )
  }).pipe(
    Effect.catchAll((error) =>
      allowUnavailable ? Effect.succeed(absolute) : Effect.fail(error),
    ),
  )
}

export const ProjectRegistryLive = Layer.effect(
  ProjectRegistry,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const sourceMutation = yield* Effect.makeSemaphore(1)

    const persistedProjects = (yield* storage.projects.get).projects
    const projectIds = new Set<ProjectId>()
    const projectSources = new Set<string>()
    for (const project of persistedProjects) {
      if (projectIds.has(project.projectId)) {
        return yield* new ProjectOperationFailed({
          operation: "validate project state",
          reason: `Duplicate project identity ${project.projectId}`,
        })
      }
      if (projectSources.has(project.sourceDirectory)) {
        return yield* new ProjectOperationFailed({
          operation: "validate project state",
          reason: `Duplicate project source ${project.sourceDirectory}`,
        })
      }
      projectIds.add(project.projectId)
      projectSources.add(project.sourceDirectory)
    }

    const list = (includeRemoved = false) =>
      storage.projects.get.pipe(
        Effect.map((state) =>
          includeRemoved
            ? state.projects
            : state.projects.filter((project) => project.registrationState === "active"),
        ),
      )

    const get = Effect.fn("acn.project-registry.get")(function* (projectId: ProjectId) {
      const project = (yield* storage.projects.get).projects.find(
        (candidate) => candidate.projectId === projectId,
      )
      return project ?? (yield* new ProjectNotFound({ projectId }))
    })

    const ensureInternalUnlocked = Effect.fn("acn.project-registry.ensure-internal")(
      function* (sourceDirectory: string, allowUnavailable: boolean) {
        const requestedSource = path.resolve(sourceDirectory)
        const existingByReference = (yield* storage.projects.get.pipe(
          Effect.mapError(operationFailed("read projects before ensuring source")),
        )).projects.find((project) => project.sourceDirectory === requestedSource)
        if (existingByReference) return existingByReference
        const source = yield* canonicalSource(fs, path, sourceDirectory, allowUnavailable)
        const now = Date.now()
        return yield* storage.projects.modify<ProjectRecord>((state) => {
          const existing = state.projects.find(
            (project) => project.sourceDirectory === source,
          )
          if (existing) {
            return [existing, state] as const
          }
          const project: ProjectRecord = {
            projectId: ProjectIdSchema.make(createId()),
            name: path.basename(source) || source,
            sourceDirectory: source,
            registrationState: "active",
            createdAt: now,
            updatedAt: now,
          }
          return [project, { projects: [...state.projects, project] }] as const
        }).pipe(Effect.mapError(operationFailed("ensure project")))
      },
    )
    const ensureInternal = (sourceDirectory: string, allowUnavailable: boolean) =>
      sourceMutation.withPermits(1)(
        ensureInternalUnlocked(sourceDirectory, allowUnavailable),
      )

    const migrationRecords = yield* storage.sessions.listProjectMigrationRecords().pipe(
      Effect.mapError(operationFailed("read session project migration records")),
    )
    const knownProjectIds = new Set(projectIds)
    for (const record of migrationRecords) {
      if (Option.isSome(record.projectId)) {
        if (knownProjectIds.has(record.projectId.value)) continue
        return yield* new ProjectOperationFailed({
          operation: "migrate session projects",
          reason: `Session ${record.sessionId} references missing project ${record.projectId.value}`,
        })
      }
      const project = yield* ensureInternal(record.workingDirectory, true)
      knownProjectIds.add(project.projectId)
      yield* storage.sessions.assignProjectId(record.sessionId, project.projectId).pipe(
        Effect.mapError(operationFailed(`assign project to session ${record.sessionId}`)),
      )
    }

    return ProjectRegistry.of({
      list,
      get,
      resolveSourceDirectory: (projectId) =>
        sourceMutation.withPermits(1)(
          get(projectId).pipe(Effect.map((project) => project.sourceDirectory)),
        ),
      ensureForSourceDirectory: (sourceDirectory, options) =>
        ensureInternal(sourceDirectory, options?.allowUnavailable ?? false),
      create: Effect.fn("acn.project-registry.create")(function (input) {
        return sourceMutation.withPermits(1)(
          Effect.gen(function* () {
            const name = yield* normalizeName(input.name)
            const sourceDirectory = yield* canonicalSource(
              fs,
              path,
              input.sourceDirectory,
              false,
            )
            const now = Date.now()
            return yield* storage.projects.modify<ProjectRecord>((state) => {
              const existing = state.projects.find(
                (project) => project.sourceDirectory === sourceDirectory,
              )
              if (existing?.registrationState === "active") {
                return [existing, state] as const
              }
              if (existing) {
                const restored: ProjectRecord = {
                  ...existing,
                  name,
                  registrationState: "active",
                  updatedAt: now,
                }
                return [restored, {
                  projects: state.projects.map((project) =>
                    project.projectId === restored.projectId ? restored : project,
                  ),
                }] as const
              }
              const created: ProjectRecord = {
                projectId: ProjectIdSchema.make(createId()),
                name,
                sourceDirectory,
                registrationState: "active",
                createdAt: now,
                updatedAt: now,
              }
              return [created, { projects: [...state.projects, created] }] as const
            }).pipe(Effect.mapError(operationFailed("create project")))
          }),
        )
      }),
      edit: Effect.fn("acn.project-registry.edit")(function (input, prepareSourceRebind) {
        return sourceMutation.withPermits(1)(
          Effect.gen(function* () {
            const name = yield* normalizeName(input.name)
            const sourceDirectory = yield* canonicalSource(
              fs,
              path,
              input.sourceDirectory,
              false,
            )
            const now = Date.now()
            const state = yield* storage.projects.get.pipe(
              Effect.mapError(operationFailed("read project before editing")),
            )
            const current = state.projects.find(
              (project) => project.projectId === input.projectId,
            )
            if (!current) return yield* new ProjectNotFound({ projectId: input.projectId })
            const duplicate = state.projects.find(
              (project) =>
                project.projectId !== input.projectId &&
                project.sourceDirectory === sourceDirectory,
            )
            if (duplicate) {
              return yield* new ProjectSourceAlreadyRegistered({
                projectId: duplicate.projectId,
                path: sourceDirectory,
              })
            }
            if (current.sourceDirectory !== sourceDirectory) {
              yield* prepareSourceRebind(current, sourceDirectory)
            }
            return yield* storage.projects.modify<
              ProjectRecord | ProjectNotFound | ProjectSourceAlreadyRegistered
            >((state) => {
              const currentAtCommit = state.projects.find(
                (project) => project.projectId === input.projectId,
              )
              if (!currentAtCommit) {
                return [new ProjectNotFound({ projectId: input.projectId }), state] as const
              }
              const duplicate = state.projects.find(
                (project) =>
                  project.projectId !== input.projectId &&
                  project.sourceDirectory === sourceDirectory,
              )
              if (duplicate) {
                return [
                  new ProjectSourceAlreadyRegistered({
                    projectId: duplicate.projectId,
                    path: sourceDirectory,
                  }),
                  state,
                ] as const
              }
              if (
                currentAtCommit.name === name &&
                currentAtCommit.sourceDirectory === sourceDirectory
              ) {
                return [currentAtCommit, state] as const
              }
              const edited: ProjectRecord = {
                ...currentAtCommit,
                name,
                sourceDirectory,
                updatedAt: now,
              }
              return [edited, {
                projects: state.projects.map((project) =>
                  project.projectId === edited.projectId ? edited : project,
                ),
              }] as const
            }).pipe(
              Effect.mapError(operationFailed("edit project")),
              Effect.flatMap((result) =>
                result instanceof ProjectNotFound ||
                result instanceof ProjectSourceAlreadyRegistered
                  ? Effect.fail(result)
                  : Effect.succeed(result),
              ),
            )
          }),
        )
      }),
      remove: Effect.fn("acn.project-registry.remove")(function* (projectId) {
        const now = Date.now()
        return yield* storage.projects.modify<ProjectRecord | ProjectNotFound>((state) => {
          const current = state.projects.find((project) => project.projectId === projectId)
          if (!current) return [new ProjectNotFound({ projectId }), state] as const
          if (current.registrationState === "removed") return [current, state] as const
          const removed: ProjectRecord = {
            ...current,
            registrationState: "removed",
            updatedAt: now,
          }
          return [removed, {
            projects: state.projects.map((project) =>
              project.projectId === projectId ? removed : project,
            ),
          }] as const
        }).pipe(
          Effect.mapError(operationFailed("remove project")),
          Effect.flatMap((result) =>
            result instanceof ProjectNotFound ? Effect.fail(result) : Effect.succeed(result),
          ),
        )
      }),
      restore: Effect.fn("acn.project-registry.restore")(function* (projectId) {
        const now = Date.now()
        return yield* storage.projects.modify<ProjectRecord | ProjectNotFound>((state) => {
          const current = state.projects.find((project) => project.projectId === projectId)
          if (!current) return [new ProjectNotFound({ projectId }), state] as const
          if (current.registrationState === "active") return [current, state] as const
          const restored: ProjectRecord = {
            ...current,
            registrationState: "active",
            updatedAt: now,
          }
          return [restored, {
            projects: state.projects.map((project) =>
              project.projectId === projectId ? restored : project,
            ),
          }] as const
        }).pipe(
          Effect.mapError(operationFailed("restore project")),
          Effect.flatMap((result) =>
            result instanceof ProjectNotFound ? Effect.fail(result) : Effect.succeed(result),
          ),
        )
      }),
      changes: storage.projects.changes.pipe(Stream.map(() => undefined)),
    })
  }),
)
