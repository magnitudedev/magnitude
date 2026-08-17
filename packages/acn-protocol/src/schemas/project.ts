import { Schema } from "effect"

export const ProjectIdSchema = Schema.NonEmptyString.pipe(Schema.brand("ProjectId"))
export type ProjectId = typeof ProjectIdSchema.Type

export const ProjectRegistrationStateSchema = Schema.Literal("active", "removed")
export type ProjectRegistrationState = typeof ProjectRegistrationStateSchema.Type

export const ProjectRecordSchema = Schema.Struct({
  projectId: ProjectIdSchema,
  name: Schema.NonEmptyString,
  sourceDirectory: Schema.NonEmptyString,
  registrationState: ProjectRegistrationStateSchema,
  createdAt: Schema.Number,
  updatedAt: Schema.Number,
})
export type ProjectRecord = typeof ProjectRecordSchema.Type

export const ProjectStateSchema = Schema.Struct({
  projects: Schema.Array(ProjectRecordSchema),
})
export type ProjectState = typeof ProjectStateSchema.Type

export const ProjectDirectoryStateSchema = Schema.Union(
  Schema.TaggedStruct("available", {}),
  Schema.TaggedStruct("missing", {}),
  Schema.TaggedStruct("inaccessible", { message: Schema.String }),
)
export type ProjectDirectoryState = typeof ProjectDirectoryStateSchema.Type

export const ProjectGitHeadSchema = Schema.Union(
  Schema.TaggedStruct("branch", { name: Schema.NonEmptyString }),
  Schema.TaggedStruct("detached", { revision: Schema.NonEmptyString }),
)
export type ProjectGitHead = typeof ProjectGitHeadSchema.Type

export const ProjectGitStateSchema = Schema.Union(
  Schema.TaggedStruct("not_repository", {}),
  Schema.TaggedStruct("repository", {
    rootDirectory: Schema.NonEmptyString,
    head: ProjectGitHeadSchema,
  }),
  Schema.TaggedStruct("unavailable", { message: Schema.String }),
)
export type ProjectGitState = typeof ProjectGitStateSchema.Type

export const ProjectSummarySchema = Schema.Struct({
  project: ProjectRecordSchema,
  directoryState: ProjectDirectoryStateSchema,
  gitState: ProjectGitStateSchema,
  openSessionCount: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  totalSessionCount: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  recentActivityAt: Schema.Number,
})
export type ProjectSummary = typeof ProjectSummarySchema.Type

export const ListProjectsResultSchema = Schema.Struct({
  projects: Schema.Array(ProjectSummarySchema),
  revealKind: Schema.Literal("finder", "folder", "unsupported"),
})
export type ListProjectsResult = typeof ListProjectsResultSchema.Type

export const ProjectChangeSchema = Schema.Struct({})
export type ProjectChange = typeof ProjectChangeSchema.Type
