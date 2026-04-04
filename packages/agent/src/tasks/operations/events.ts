import type { TaskAssigned, TaskCancelled, TaskCreated, TaskUpdated } from '../../events'

declare const ValidatedBrand: unique symbol
export type Validated<T> = T & { readonly [ValidatedBrand]: true }

export type ValidatedTaskGraphEvent =
  | Validated<TaskCreated>
  | Validated<TaskUpdated>
  | Validated<TaskAssigned>
  | Validated<TaskCancelled>

export type ValidatedTaskEvent = ValidatedTaskGraphEvent
