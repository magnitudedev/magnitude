import { Schema } from "effect"
import { DirectoryPathSchema } from "./paths"

export const ProjectIdSchema = Schema.NonEmptyString.pipe(Schema.brand("ProjectId"))
export type ProjectId = typeof ProjectIdSchema.Type

export const ProjectRegistrationStateSchema = Schema.Literal("active", "removed")
export type ProjectRegistrationState = typeof ProjectRegistrationStateSchema.Type

export const ProjectSchema = Schema.Struct({
  projectId: ProjectIdSchema,
  name: Schema.NonEmptyString,
  cwd: DirectoryPathSchema,
  registrationState: ProjectRegistrationStateSchema,
  createdAt: Schema.Number,
  updatedAt: Schema.Number,
})
export type Project = typeof ProjectSchema.Type

export const ProjectStateSchema = Schema.Struct({
  projects: Schema.Array(ProjectSchema),
})
export type ProjectState = typeof ProjectStateSchema.Type

export const DirectoryInspectionSchema = Schema.Union(
  Schema.TaggedStruct("available", {}),
  Schema.TaggedStruct("missing", {}),
  Schema.TaggedStruct("access_denied", {}),
  Schema.TaggedStruct("not_directory", {}),
  Schema.TaggedStruct("unavailable", {}),
)
export type DirectoryInspection = typeof DirectoryInspectionSchema.Type

export const ProjectGitHeadSchema = Schema.Union(
  Schema.TaggedStruct("branch", { name: Schema.NonEmptyString }),
  Schema.TaggedStruct("detached", { revision: Schema.NonEmptyString }),
)
export type ProjectGitHead = typeof ProjectGitHeadSchema.Type

export const GitInspectionSchema = Schema.Union(
  Schema.TaggedStruct("git_unavailable", {}),
  Schema.TaggedStruct("not_git_repository", {}),
  Schema.TaggedStruct("git_repository", {
    rootDirectory: DirectoryPathSchema,
    head: ProjectGitHeadSchema,
  }),
  Schema.TaggedStruct("git_inspection_failed", {}),
)
export type GitInspection = typeof GitInspectionSchema.Type

export const ProjectPageCursorSchema = Schema.String.pipe(Schema.brand("ProjectPageCursor"))
export type ProjectPageCursor = typeof ProjectPageCursorSchema.Type

export const ProjectPageRequestSchema = Schema.Struct({
  includeRemoved: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  cursor: Schema.optionalWith(ProjectPageCursorSchema, { as: "Option", exact: true }),
  limit: Schema.optionalWith(Schema.Number.pipe(Schema.int(), Schema.between(1, 100)), {
    default: () => 20,
  }),
})
export type ProjectPageRequest = typeof ProjectPageRequestSchema.Type

export const ProjectPageSchema = Schema.Struct({
  items: Schema.Array(ProjectSchema),
  nextCursor: Schema.optionalWith(ProjectPageCursorSchema, { as: "Option", exact: true }),
})
export type ProjectPage = typeof ProjectPageSchema.Type

export const ProjectInspectionSchema = Schema.Struct({
  projectId: ProjectIdSchema,
  directory: DirectoryInspectionSchema,
  git: GitInspectionSchema,
})
export type ProjectInspection = typeof ProjectInspectionSchema.Type

export const ProjectChangeSchema = Schema.Struct({})
export type ProjectChange = typeof ProjectChangeSchema.Type
