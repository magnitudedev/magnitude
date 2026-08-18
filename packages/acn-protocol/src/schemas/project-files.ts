import { Schema } from "effect"
import { ProjectIdSchema } from "./project"

const isProjectRelativePath = (value: string): boolean => {
  if (value === "") return true
  if (value.includes("\0") || value.includes("\\") || value.startsWith("/")) return false
  if (/^[A-Za-z]:/.test(value)) return false
  return value.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..")
}

export const ProjectRelativePathSchema = Schema.String.pipe(
  Schema.filter(isProjectRelativePath, { message: () => "Expected a normalized project-relative path" }),
  Schema.brand("ProjectRelativePath"),
)
export type ProjectRelativePath = Schema.Schema.Type<typeof ProjectRelativePathSchema>

export const ProjectFileRevisionSchema = Schema.String.pipe(Schema.brand("ProjectFileRevision"))
export type ProjectFileRevision = Schema.Schema.Type<typeof ProjectFileRevisionSchema>

export const ProjectDirectoryEntrySchema = Schema.Struct({
  name: Schema.String,
  path: ProjectRelativePathSchema,
  kind: Schema.Literal("directory", "file"),
  size: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
})
export type ProjectDirectoryEntry = Schema.Schema.Type<typeof ProjectDirectoryEntrySchema>

export const ProjectDirectoryListingSchema = Schema.Struct({
  projectId: ProjectIdSchema,
  directory: ProjectRelativePathSchema,
  entries: Schema.Array(ProjectDirectoryEntrySchema),
})
export type ProjectDirectoryListing = Schema.Schema.Type<typeof ProjectDirectoryListingSchema>

export const ProjectFilesChangeSchema = Schema.Struct({
  projectId: ProjectIdSchema,
})
export type ProjectFilesChange = Schema.Schema.Type<typeof ProjectFilesChangeSchema>

export const ProjectEntryMoveSchema = Schema.Struct({
  sourcePath: ProjectRelativePathSchema,
  destinationPath: ProjectRelativePathSchema,
  kind: Schema.Literal("directory", "file"),
})
export type ProjectEntryMove = Schema.Schema.Type<typeof ProjectEntryMoveSchema>

export const ProjectFileTextSnapshotSchema = Schema.TaggedStruct("text", {
  path: ProjectRelativePathSchema,
  content: Schema.String,
  revision: ProjectFileRevisionSchema,
  size: Schema.Number,
  newline: Schema.Literal("lf", "crlf", "mixed", "none"),
})
export type ProjectFileTextSnapshot = Schema.Schema.Type<typeof ProjectFileTextSnapshotSchema>

export const ProjectFileImageSnapshotSchema = Schema.TaggedStruct("image", {
  path: ProjectRelativePathSchema,
  mediaType: Schema.Literal("image/png", "image/jpeg", "image/gif", "image/webp"),
  data: Schema.String,
  revision: ProjectFileRevisionSchema,
  size: Schema.Number,
})

export const ProjectFileUnsupportedSnapshotSchema = Schema.TaggedStruct("unsupported", {
  path: ProjectRelativePathSchema,
  reason: Schema.Literal("binary", "too_large", "unsupported_type"),
  size: Schema.Number,
})

export const ProjectFileSnapshotSchema = Schema.Union(
  ProjectFileTextSnapshotSchema,
  ProjectFileImageSnapshotSchema,
  ProjectFileUnsupportedSnapshotSchema,
)
export type ProjectFileSnapshot = Schema.Schema.Type<typeof ProjectFileSnapshotSchema>
