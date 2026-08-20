import { Schema } from "effect"
import { ProjectIdSchema } from "./project"
import { RelativePathSchema } from "./paths"

/**
 * Content digest of one project file's bytes, used to prevent stale writes.
 * It is not a Project revision and not a version of the Project record.
 */
export const FileContentHashSchema = Schema.String.pipe(Schema.brand("FileContentHash"))
export type FileContentHash = Schema.Schema.Type<typeof FileContentHashSchema>

export const ProjectDirectoryEntrySchema = Schema.Struct({
  name: Schema.String,
  path: RelativePathSchema,
  kind: Schema.Literal("directory", "file"),
  size: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
})
export type ProjectDirectoryEntry = Schema.Schema.Type<typeof ProjectDirectoryEntrySchema>

export const ProjectDirectoryListingSchema = Schema.Struct({
  projectId: ProjectIdSchema,
  directory: RelativePathSchema,
  entries: Schema.Array(ProjectDirectoryEntrySchema),
})
export type ProjectDirectoryListing = Schema.Schema.Type<typeof ProjectDirectoryListingSchema>

export const ProjectFilesChangeSchema = Schema.Struct({
  projectId: ProjectIdSchema,
})
export type ProjectFilesChange = Schema.Schema.Type<typeof ProjectFilesChangeSchema>

export const ProjectEntryMoveSchema = Schema.Struct({
  sourcePath: RelativePathSchema,
  destinationPath: RelativePathSchema,
  kind: Schema.Literal("directory", "file"),
})
export type ProjectEntryMove = Schema.Schema.Type<typeof ProjectEntryMoveSchema>

export const ProjectFileTextSnapshotSchema = Schema.TaggedStruct("text", {
  path: RelativePathSchema,
  content: Schema.String,
  contentHash: FileContentHashSchema,
  size: Schema.Number,
  newline: Schema.Literal("lf", "crlf", "mixed", "none"),
})
export type ProjectFileTextSnapshot = Schema.Schema.Type<typeof ProjectFileTextSnapshotSchema>

export const ProjectFileImageSnapshotSchema = Schema.TaggedStruct("image", {
  path: RelativePathSchema,
  mediaType: Schema.Literal("image/png", "image/jpeg", "image/gif", "image/webp"),
  data: Schema.String,
  contentHash: FileContentHashSchema,
  size: Schema.Number,
})

export const ProjectFileUnsupportedSnapshotSchema = Schema.TaggedStruct("unsupported", {
  path: RelativePathSchema,
  reason: Schema.Literal("binary", "too_large", "unsupported_type"),
  size: Schema.Number,
})

export const ProjectFileSnapshotSchema = Schema.Union(
  ProjectFileTextSnapshotSchema,
  ProjectFileImageSnapshotSchema,
  ProjectFileUnsupportedSnapshotSchema,
)
export type ProjectFileSnapshot = Schema.Schema.Type<typeof ProjectFileSnapshotSchema>
