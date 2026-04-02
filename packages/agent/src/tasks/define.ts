import type { TaskTypeDefinition } from './types'

export function defineTaskType<const TId extends string>(
  definition: TaskTypeDefinition<TId>,
): TaskTypeDefinition<TId> {
  return definition
}
