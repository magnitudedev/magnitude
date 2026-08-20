import {
  type ProjectId,
  type ProjectInspection,
  type ProjectNotFound,
  type ProjectStoreUnavailable,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Layer } from "effect"
import { FileSystemManager } from "./file-system-manager"
import { GitInspector } from "./git-inspector"
import { ProjectStore } from "./project-store"

export interface ProjectInspector {
  readonly inspect: (
    projectId: ProjectId,
  ) => Effect.Effect<ProjectInspection, ProjectNotFound | ProjectStoreUnavailable>
}

export const ProjectInspector = Context.GenericTag<ProjectInspector>("acn/ProjectInspector")

export const ProjectInspectorLive: Layer.Layer<
  ProjectInspector,
  never,
  ProjectStore | FileSystemManager | GitInspector
> = Layer.effect(
  ProjectInspector,
  Effect.gen(function* () {
    const store = yield* ProjectStore
    const fileSystem = yield* FileSystemManager
    const git = yield* GitInspector

    return ProjectInspector.of({
      inspect: Effect.fn("acn.project-inspector.inspect")(function* (projectId) {
        const project = yield* store.get(projectId)
        const directory = yield* fileSystem.inspectDirectory(project.cwd)
        return {
          projectId,
          directory,
          git: directory._tag === "available"
            ? yield* git.inspect(project.cwd)
            : { _tag: "git_inspection_failed" },
        }
      }),
    })
  }),
)
