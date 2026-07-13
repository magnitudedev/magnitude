import { BaseStateSchema } from '@magnitudedev/harness'
export type { BaseState, Phase } from '@magnitudedev/harness'
import {
  defineJsonEncodedSchemaEntries,
  makeSchemaUnionFromEntries,
  schemaMapFromEntries,
} from '@magnitudedev/utils/schema'
import { CheckpointChangesStateSchema, CheckpointRollbackStateSchema } from '@magnitudedev/vcs'
import { Schema } from 'effect'
import { EditDiffSchema } from './edit-diff'

export const CompactStateSchema = BaseStateSchema

export const CreateTaskStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const FileReadStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  lineCount: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  offset: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  limit: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const FileWriteStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  body: Schema.String,
  charCount: Schema.Number,
  lineCount: Schema.Number,
  isScratchpad: Schema.Boolean,
  scratchpadDisplayPath: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const FileEditStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  oldText: Schema.String,
  newText: Schema.String,
  replaceAll: Schema.Boolean,
  streamingTarget: Schema.optionalWith(Schema.Literal('old', 'new'), { as: 'Option', exact: true }),
  baseContent: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  diffs: Schema.Array(EditDiffSchema),
  isScratchpad: Schema.Boolean,
  scratchpadDisplayPath: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const TreeEntrySchema = Schema.Struct({
  path: Schema.String,
  name: Schema.String,
  type: Schema.Literal('file', 'dir'),
  depth: Schema.Number,
})

export const FileTreeStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  entries: Schema.Array(TreeEntrySchema),
  fileCount: Schema.Number,
  dirCount: Schema.Number,
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const SearchMatchSchema = Schema.Struct({
  file: Schema.String,
  match: Schema.String,
})

export const FileSearchStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  pattern: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  glob: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  limit: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  matches: Schema.Array(SearchMatchSchema),
  matchCount: Schema.Number,
  fileCount: Schema.Number,
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const FileViewStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const QueryImageStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  query: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const ShellStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  command: Schema.String,
  done: Schema.optionalWith(Schema.Literal('completed', 'detached'), { as: 'Option', exact: true }),
  exitCode: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  stdout: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  stderr: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  stdoutPath: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  stderrPath: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  pid: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  partialStdout: Schema.String,
  partialStderr: Schema.String,
}))

export const WebSearchSourceSchema = Schema.Struct({
  title: Schema.String,
  url: Schema.String,
})

export const WebSearchStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  query: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  sources: Schema.optionalWith(Schema.Array(WebSearchSourceSchema), { as: 'Option', exact: true }),
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const WebFetchStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  url: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const UpdateTaskStatusSchema = Schema.Literal('pending', 'completed', 'cancelled')

export const UpdateTaskStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  status: Schema.optionalWith(UpdateTaskStatusSchema, { as: 'Option', exact: true }),
}))

export const SpawnWorkerStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  role: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  agentId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  message: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  yield: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
  title: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const KillWorkerStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const ReassignWorkerStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  agentId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const MessageWorkerStateSchema = BaseStateSchema

export const SkillActivationStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  skillName: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  skillPath: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  contentPreview: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  errorDetail: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const MessageAdvisorStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  message: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  responsePreview: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const FinishGoalStateSchema = Schema.extend(BaseStateSchema, Schema.Struct({
  evidence: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  goalId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
}))

export const ToolStateSchemaEntries = defineJsonEncodedSchemaEntries([
  ['fileRead', FileReadStateSchema],
  ['fileWrite', FileWriteStateSchema],
  ['fileEdit', FileEditStateSchema],
  ['fileTree', FileTreeStateSchema],
  ['fileSearch', FileSearchStateSchema],
  ['fileView', FileViewStateSchema],
  ['queryImage', QueryImageStateSchema],
  ['shell', ShellStateSchema],
  ['webSearch', WebSearchStateSchema],
  ['webFetch', WebFetchStateSchema],
  ['createTask', CreateTaskStateSchema],
  ['updateTask', UpdateTaskStateSchema],
  ['spawnWorker', SpawnWorkerStateSchema],
  ['killWorker', KillWorkerStateSchema],
  ['reassignWorker', ReassignWorkerStateSchema],
  ['messageWorker', MessageWorkerStateSchema],
  ['skill', SkillActivationStateSchema],
  ['messageAdvisor', MessageAdvisorStateSchema],
  ['finishGoal', FinishGoalStateSchema],
  ['compact', CompactStateSchema],
  ['checkpointRollback', CheckpointRollbackStateSchema],
  ['checkpointChanges', CheckpointChangesStateSchema],
] as const)

export const ToolStateSchemaByKey = schemaMapFromEntries(ToolStateSchemaEntries)
export type ToolStateSchemaByKey = typeof ToolStateSchemaByKey
export type ToolKey = keyof ToolStateSchemaByKey

export const ToolStateSchema = makeSchemaUnionFromEntries(ToolStateSchemaEntries)

export type CompactState = typeof CompactStateSchema.Type
export type CreateTaskState = typeof CreateTaskStateSchema.Type
export type FileReadState = typeof FileReadStateSchema.Type
export type FileWriteState = typeof FileWriteStateSchema.Type
export type FileEditState = typeof FileEditStateSchema.Type
export type TreeEntry = typeof TreeEntrySchema.Type
export type FileTreeState = typeof FileTreeStateSchema.Type
export type SearchMatch = typeof SearchMatchSchema.Type
export type FileSearchState = typeof FileSearchStateSchema.Type
export type FileViewState = typeof FileViewStateSchema.Type
export type QueryImageState = typeof QueryImageStateSchema.Type
export type ShellState = typeof ShellStateSchema.Type
export type WebSearchSource = typeof WebSearchSourceSchema.Type
export type WebSearchState = typeof WebSearchStateSchema.Type
export type WebFetchState = typeof WebFetchStateSchema.Type
export type UpdateTaskStatus = typeof UpdateTaskStatusSchema.Type
export type UpdateTaskState = typeof UpdateTaskStateSchema.Type
export type SpawnWorkerState = typeof SpawnWorkerStateSchema.Type
export type KillWorkerState = typeof KillWorkerStateSchema.Type
export type ReassignWorkerState = typeof ReassignWorkerStateSchema.Type
export type MessageWorkerState = typeof MessageWorkerStateSchema.Type
export type SkillActivationState = typeof SkillActivationStateSchema.Type
export type MessageAdvisorState = typeof MessageAdvisorStateSchema.Type
export type FinishGoalState = typeof FinishGoalStateSchema.Type
export type CheckpointRollbackState = typeof CheckpointRollbackStateSchema.Type
export type CheckpointChangesState = typeof CheckpointChangesStateSchema.Type
export type ToolState = typeof ToolStateSchema.Type
export type ToolStateFromSchema = ToolState

export type ContentState = FileWriteState
export type DiffState = FileEditState

export function isToolKey(value: string): value is ToolKey {
  return value in ToolStateSchemaByKey
}

export function decodeToolVisualState(toolKey: string, value: unknown): ToolState | null {
  if (!isToolKey(toolKey)) return null
  const schema = ToolStateSchemaByKey[toolKey] as Schema.Schema.AnyNoContext
  const decoded = Schema.decodeUnknownOption(schema)(value)
  return decoded._tag === 'Some' ? decoded.value as ToolState : null
}
