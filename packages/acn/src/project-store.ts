import {
  InvalidProjectPageCursor,
  ProjectCwdAlreadyRegistered,
  ProjectIdSchema,
  ProjectNotFound,
  ProjectPageCursorSchema,
  ProjectStoreUnavailable,
  type DirectoryPath,
  type Project,
  type ProjectId,
  type ProjectPage,
  type ProjectPageCursor,
  type ProjectPageRequest,
  type ProjectState,
} from "@magnitudedev/acn-protocol"
import { MagnitudeStorage, type StateDocumentError } from "@magnitudedev/storage"
import { Context, Effect, Either, Layer, Option, Schema, Stream } from "effect"

export interface ProjectStore {
  readonly get: (
    projectId: ProjectId,
  ) => Effect.Effect<Project, ProjectNotFound | ProjectStoreUnavailable>
  readonly findByCwd: (
    cwd: DirectoryPath,
    includeRemoved?: boolean,
  ) => Effect.Effect<Option.Option<Project>, ProjectStoreUnavailable>
  readonly page: (
    request: ProjectPageRequest,
  ) => Effect.Effect<ProjectPage, InvalidProjectPageCursor | ProjectStoreUnavailable>
  readonly insert: (
    project: Project,
  ) => Effect.Effect<Project, ProjectCwdAlreadyRegistered | ProjectStoreUnavailable>
  readonly update: (
    projectId: ProjectId,
    change: (current: Project) => Project,
  ) => Effect.Effect<Project, ProjectNotFound | ProjectCwdAlreadyRegistered | ProjectStoreUnavailable>
  readonly changes: Stream.Stream<void>
}

export const ProjectStore = Context.GenericTag<ProjectStore>("acn/ProjectStore")

const ProjectCursorContent = Schema.compose(
  Schema.StringFromBase64Url,
  Schema.parseJson(Schema.Struct({
    updatedAt: Schema.Number,
    projectId: ProjectIdSchema,
  })),
)

const decodeCursor = (cursor: ProjectPageCursor) =>
  Schema.decodeUnknown(ProjectCursorContent)(cursor).pipe(
    Effect.mapError(() => new InvalidProjectPageCursor()),
  )

const encodeCursor = (project: Project): ProjectPageCursor =>
  ProjectPageCursorSchema.make(Schema.encodeSync(ProjectCursorContent)({
    updatedAt: project.updatedAt,
    projectId: project.projectId,
  }))

const byRecency = (left: Project, right: Project): number =>
  right.updatedAt - left.updatedAt || right.projectId.localeCompare(left.projectId)

const strictlyAfter = (
  project: Project,
  cursor: { readonly updatedAt: number; readonly projectId: string },
): boolean =>
  project.updatedAt < cursor.updatedAt
  || (project.updatedAt === cursor.updatedAt && project.projectId < cursor.projectId)

const storeUnavailable = <A, R>(
  effect: Effect.Effect<A, StateDocumentError, R>,
): Effect.Effect<A, ProjectStoreUnavailable, R> =>
  effect.pipe(
    Effect.tapError((error) => Effect.logWarning("Project state document unavailable").pipe(
      Effect.annotateLogs({ errorTag: error._tag }),
    )),
    Effect.mapError(() => new ProjectStoreUnavailable()),
  )

export const ProjectStoreLive: Layer.Layer<ProjectStore, never, MagnitudeStorage> = Layer.effect(
  ProjectStore,
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage

    return ProjectStore.of({
      get: Effect.fn("acn.project-store.get")(function* (projectId) {
        const project = (yield* storage.projects.get).projects.find(
          (candidate) => candidate.projectId === projectId,
        )
        return project ?? (yield* new ProjectNotFound({ projectId }))
      }),
      findByCwd: (cwd, includeRemoved = false) => storage.projects.get.pipe(
        Effect.map((state) => Option.fromNullable(state.projects.find((project) =>
          project.cwd === cwd
          && (includeRemoved || project.registrationState === "active")))),
      ),
      page: Effect.fn("acn.project-store.page")(function* (request) {
        const cursor = yield* Option.match(request.cursor, {
          onNone: () => Effect.succeedNone,
          onSome: (value) => decodeCursor(value).pipe(Effect.map(Option.some)),
        })
        const ordered = (yield* storage.projects.get).projects
          .filter((project) => request.includeRemoved || project.registrationState === "active")
          .sort(byRecency)
          .filter((project) => Option.match(cursor, {
            onNone: () => true,
            onSome: (key) => strictlyAfter(project, key),
          }))
        const window = ordered.slice(0, request.limit + 1)
        const items = window.slice(0, request.limit)
        const last = items.at(-1)
        return {
          items,
          nextCursor: window.length > request.limit && last !== undefined
            ? Option.some(encodeCursor(last))
            : Option.none(),
        }
      }),
      insert: Effect.fn("acn.project-store.insert")(function* (project) {
        const result = yield* storage.projects.modify(
          (state): readonly [Either.Either<Project, ProjectCwdAlreadyRegistered>, ProjectState] => {
            const existing = state.projects.find((candidate) => candidate.cwd === project.cwd)
            if (existing !== undefined) {
              return [Either.left(new ProjectCwdAlreadyRegistered({
                projectId: existing.projectId,
                cwd: project.cwd,
              })), state]
            }
            return [Either.right(project), { projects: [...state.projects, project] }]
          },
        ).pipe(storeUnavailable)
        return yield* result
      }),
      update: Effect.fn("acn.project-store.update")(function* (projectId, change) {
        const result = yield* storage.projects.modify(
          (state): readonly [
            Either.Either<Project, ProjectNotFound | ProjectCwdAlreadyRegistered>,
            ProjectState,
          ] => {
            const current = state.projects.find((project) => project.projectId === projectId)
            if (current === undefined) {
              return [Either.left(new ProjectNotFound({ projectId })), state]
            }
            const next = change(current)
            const duplicate = state.projects.find((project) =>
              project.projectId !== projectId && project.cwd === next.cwd)
            if (duplicate !== undefined) {
              return [Either.left(new ProjectCwdAlreadyRegistered({
                projectId: duplicate.projectId,
                cwd: next.cwd,
              })), state]
            }
            return [Either.right(next), {
              projects: state.projects.map((project) =>
                project.projectId === projectId ? next : project),
            }]
          },
        ).pipe(storeUnavailable)
        return yield* result
      }),
      changes: storage.projects.changes.pipe(Stream.drop(1), Stream.as<void>(undefined)),
    })
  }),
)
