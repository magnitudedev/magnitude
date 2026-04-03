import { Schema } from '@effect/schema'
import type { TaskAssignee, TaskTypeDefinition, TaskTypeKind } from '../types'

const NonEmptyTrimmed = Schema.String.pipe(
  Schema.trimmed(),
  Schema.minLength(1)
)

const WorkerAssigneeSchema = Schema.Literal(
  'builder',
  'explorer',
  'planner',
  'debugger',
  'reviewer',
  'browser'
)

const TaskAssigneeSchema = Schema.Union(
  Schema.Literal('user'),
  WorkerAssigneeSchema
)

const ParsedSectionsSchema = Schema.Struct({
  lead: Schema.String,
  worker: Schema.optional(Schema.String),
  criteria: Schema.String,
})

const TaskFrontmatterSchema = Schema.Struct({
  id: NonEmptyTrimmed,
  label: NonEmptyTrimmed,
  description: NonEmptyTrimmed,
  allowedAssignees: Schema.Array(TaskAssigneeSchema),
})

export function deriveTaskKind(allowedAssignees: ReadonlyArray<TaskAssignee>): TaskTypeKind {
  if (allowedAssignees.length === 0) return 'composite'
  const hasUser = allowedAssignees.includes('user')
  const hasWorkerRole = allowedAssignees.some((a) => a !== 'user')

  if (hasUser && hasWorkerRole) return 'generic'
  if (hasUser) return 'user'
  return 'leaf'
}

const RawParsedDefinitionSchema = Schema.Struct({
  frontmatter: TaskFrontmatterSchema,
  sections: ParsedSectionsSchema,
})

export const TaskDefinitionSchema = RawParsedDefinitionSchema.pipe(
  Schema.transform(
    Schema.Unknown as Schema.Schema<TaskTypeDefinition>,
    {
      strict: true,
      decode: ({ frontmatter, sections }) => {
        const kind = deriveTaskKind(frontmatter.allowedAssignees as ReadonlyArray<TaskAssignee>)

        if (kind === 'leaf') {
          if (!sections.worker) {
            throw new Error('Leaf task definitions must include <!-- @worker --> section.')
          }
          return {
            ...frontmatter,
            allowedAssignees: frontmatter.allowedAssignees as TaskTypeDefinition['allowedAssignees'],
            kind,
            leadGuidance: sections.lead,
            workerGuidance: sections.worker,
            criteria: sections.criteria,
          } as TaskTypeDefinition
        }

        if (kind === 'generic') {
          return {
            ...frontmatter,
            allowedAssignees: frontmatter.allowedAssignees as TaskTypeDefinition['allowedAssignees'],
            kind,
            leadGuidance: sections.lead,
            workerGuidance: sections.worker,
            criteria: sections.criteria,
          } as TaskTypeDefinition
        }

        if (sections.worker) {
          throw new Error(`Task kind "${kind}" must not include <!-- @worker --> section.`)
        }

        return {
          ...frontmatter,
          allowedAssignees: frontmatter.allowedAssignees as TaskTypeDefinition['allowedAssignees'],
          kind,
          leadGuidance: sections.lead,
          criteria: sections.criteria,
        } as TaskTypeDefinition
      },
      encode: (taskType) => taskType as never,
    }
  )
)
