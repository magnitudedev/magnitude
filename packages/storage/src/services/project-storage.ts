import { Context, Layer } from 'effect'

import {
  makeProjectStoragePaths,
  type ProjectStoragePaths,
} from '../paths'

export interface ProjectStorageShape {
  readonly cwd: string
  readonly root: string
  readonly paths: ProjectStoragePaths
}

export class ProjectStorage extends Context.Tag('ProjectStorage')<
  ProjectStorage,
  ProjectStorageShape
>() {}

export function makeProjectStorage(options: {
  readonly cwd: string
}): ProjectStorageShape {
  const paths = makeProjectStoragePaths(options.cwd)

  return {
    cwd: options.cwd,
    root: paths.root,
    paths,
  }
}

export function ProjectStorageLiveFromCwd(
  cwd: string
): Layer.Layer<ProjectStorage> {
  return Layer.succeed(ProjectStorage, ProjectStorage.of(makeProjectStorage({ cwd })))
}

export const ProjectStorageLiveFromProcessCwd = ProjectStorageLiveFromCwd(
  process.cwd()
)