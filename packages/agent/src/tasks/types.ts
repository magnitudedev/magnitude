import type { AgentVariant } from '../agents'

export type TaskAssignee = 'self' | 'user' | AgentVariant

export interface TaskTypeDefinition<TId extends string = string> {
  readonly id: TId
  readonly label: string
  readonly description: string
  readonly allowedAssignees: readonly TaskAssignee[]
  readonly strategy: string
}
