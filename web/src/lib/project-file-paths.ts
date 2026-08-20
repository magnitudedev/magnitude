import { RelativePathSchema, type RelativePath } from "@magnitudedev/sdk"

export const parentProjectPath = (path: RelativePath): RelativePath => {
  const separator = path.lastIndexOf("/")
  return RelativePathSchema.make(separator === -1 ? "" : path.slice(0, separator))
}

export const isProjectPathWithin = (
  path: RelativePath,
  directory: RelativePath,
): boolean => directory === "" || path === directory || path.startsWith(`${directory}/`)

export const translateProjectPath = (
  path: RelativePath,
  sourcePath: RelativePath,
  destinationPath: RelativePath,
): RelativePath => {
  if (path === sourcePath) return destinationPath
  if (!path.startsWith(`${sourcePath}/`)) return path
  return RelativePathSchema.make(`${destinationPath}${path.slice(sourcePath.length)}`)
}

export const canMoveProjectEntryToDirectory = (
  source: { readonly path: RelativePath; readonly kind: "directory" | "file" },
  destinationDirectory: RelativePath,
): boolean => parentProjectPath(source.path) !== destinationDirectory
  && !(source.kind === "directory" && isProjectPathWithin(destinationDirectory, source.path))
