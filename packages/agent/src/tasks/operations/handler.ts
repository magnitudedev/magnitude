import { Effect } from 'effect'
import type { Skill } from '@magnitudedev/skills'
import type { TaskOperationGraphSnapshot } from './types'
import { handleCreateDirective, type CreateDirective } from './create'
import { handleUpdateDirective, type UpdateDirective } from './update'
import { handleCancelDirective, type CancelDirective } from './cancel'
import { handleMessageDirective, type MessageDirective, type MessageDirectiveSuccess } from './message'
import { handleSpawnWorkerDirective, type SpawnWorkerDirective } from './spawn-worker'
import { handleKillWorkerDirective, type KillWorkerDirective } from './kill-worker'

export interface TaskDirectiveContext {
  readonly forkId: string | null
  readonly timestamp: number
  readonly graph: TaskOperationGraphSnapshot
  readonly skills: Map<string, Skill>
}

export type TaskDirectiveResult =
  | { success: true }
  | { success: true; title: string }
  | { success: false; code: string; error: string }

export type TaskDirectiveErrorResult = { success: false; code: string; error: string }

export type TaskDirective<R = never> =
  | CreateDirective
  | UpdateDirective
  | CancelDirective
  | MessageDirective
  | SpawnWorkerDirective<R>
  | KillWorkerDirective

export type HandleTaskDirectiveResult = TaskDirectiveErrorResult | { success: true } | { success: true; title: string } | MessageDirectiveSuccess

export const handleTaskDirective = <R = never>(
  directive: TaskDirective<R>,
  context: TaskDirectiveContext,
) => {
  switch (directive.kind) {
    case 'create':
      return handleCreateDirective(directive, context)
    case 'update':
      return handleUpdateDirective(directive, context)
    case 'cancel':
      return handleCancelDirective(directive, context)
    case 'message':
      return handleMessageDirective(directive, context)
    case 'spawn-worker':
      return handleSpawnWorkerDirective(directive, context)
    case 'kill-worker':
      return handleKillWorkerDirective(directive, context)
  }
}
