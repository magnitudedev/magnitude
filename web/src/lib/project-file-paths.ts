import { ProjectRelativePathSchema, type ProjectRelativePath } from "@magnitudedev/sdk"

export const parentProjectPath = (path: ProjectRelativePath): ProjectRelativePath => {
  const separator = path.lastIndexOf("/")
  return ProjectRelativePathSchema.make(separator === -1 ? "" : path.slice(0, separator))
}

export const isProjectPathWithin = (
  path: ProjectRelativePath,
  directory: ProjectRelativePath,
): boolean => directory === "" || path === directory || path.startsWith(`${directory}/`)

export const translateProjectPath = (
  path: ProjectRelativePath,
  sourcePath: ProjectRelativePath,
  destinationPath: ProjectRelativePath,
): ProjectRelativePath => {
  if (path === sourcePath) return destinationPath
  if (!path.startsWith(`${sourcePath}/`)) return path
  return ProjectRelativePathSchema.make(`${destinationPath}${path.slice(sourcePath.length)}`)
}

export const canMoveProjectEntryToDirectory = (
  source: { readonly path: ProjectRelativePath; readonly kind: "directory" | "file" },
  destinationDirectory: ProjectRelativePath,
): boolean => parentProjectPath(source.path) !== destinationDirectory
  && !(source.kind === "directory" && isProjectPathWithin(destinationDirectory, source.path))
