/**
 * Execution Types
 *
 * Types for the execution manager service interface
 * and the ExecutionManager Effect service tag.
 */

import { Effect, Context, Layer } from 'effect'
import type { AttemptCommitPolicy, TurnOutcome } from '../events'
import type { ResponseUsage } from '@magnitudedev/ai'
import type { Projection, WorkerBusService } from '@magnitudedev/event-core'
import type { RoleId } from '../agents/role-validation'
import type { AgentRoutingProjection } from '../projections/agent-routing'
import type { AgentLifecycleProjection } from '../projections/agent-lifecycle'
import type { TaskGraphProjection } from '../projections/task-graph'
import type { GoalProjection } from '../projections/goal'
import type { TurnProjection } from '../projections/turn'
import type { SessionContextProjection } from '../projections/session-context'
import type { ConversationProjection } from '../projections/conversation'
import type { WindowProjection } from '../window'
import type { ChatPersistence } from '../persistence/chat-persistence-service'
import type { BoundObservable } from '../observables/types'
import type { JsonSchema } from '../prompts/fork-context'
import type { AppEvent } from '../events'
import type { ForkLayer } from './fork-layer'


// =============================================================================
// Turn Result
// =============================================================================

/**
 * Agent-local usage type. Extends ai.ResponseUsage with optional cost fields.
 * Cost fields are nullable until a cost computation source is available.
 */
export interface AgentCallUsage {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens: number
  readonly cacheWriteTokens: number | null
  readonly totalCost: number | null
}

/** Map ai.ResponseUsage to agent usage. */
export function fromResponseUsage(usage: ResponseUsage): AgentCallUsage {
  return {
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    cacheReadTokens: usage.cacheReadTokens,
    cacheWriteTokens: null,
    totalCost: usage.cost ?? null,
  }
}

export interface TurnStrategyResult {
  readonly executeResult: ExecuteResult
  readonly usage: AgentCallUsage
}

// =============================================================================
// ExecutionManager Service
// =============================================================================

export const IDENTICAL_RESPONSE_BREAKER_THRESHOLD = 5

type SessionContextProjectionInstance = Projection.ProjectionInstance<typeof SessionContextProjection.stateSchema>
type AgentRoutingProjectionInstance = Projection.ProjectionInstance<typeof AgentRoutingProjection.stateSchema>
type AgentLifecycleProjectionInstance = Projection.ProjectionInstance<typeof AgentLifecycleProjection.stateSchema>
type TurnProjectionInstance = Projection.ForkedProjectionInstance<typeof TurnProjection.forkStateSchema>
type WindowProjectionInstance = Projection.ForkedProjectionInstance<typeof WindowProjection.forkStateSchema>
type TaskGraphProjectionInstance = Projection.ProjectionInstance<typeof TaskGraphProjection.stateSchema>
type GoalProjectionInstance = Projection.ProjectionInstance<typeof GoalProjection.stateSchema>
type ConversationProjectionInstance = Projection.ProjectionInstance<typeof ConversationProjection.stateSchema>

type ExecutionProjectionRequirements =
  | SessionContextProjectionInstance
  | AgentRoutingProjectionInstance
  | AgentLifecycleProjectionInstance
  | TurnProjectionInstance
  | WindowProjectionInstance
  | TaskGraphProjectionInstance
  | GoalProjectionInstance
  | ConversationProjectionInstance

export interface ExecuteResult {
  readonly result: TurnOutcome
  readonly commitPolicy?: AttemptCommitPolicy
  readonly usage: {
    readonly inputTokens: number | null
    readonly outputTokens: number | null
    readonly cacheReadTokens: number | null
    readonly cacheWriteTokens: number | null
    readonly cost: number | null
  } | null
}

export interface ExecutionManagerService {
  readonly initFork: (
    forkId: string | null,
    variant: RoleId
  ) => Effect.Effect<
    void,
    never,
    ExecutionProjectionRequirements | ChatPersistence | WorkerBusService<AppEvent>
  >

  readonly disposeFork: (forkId: string | null) => Effect.Effect<void>

  readonly getObservables: (forkId: string | null) => BoundObservable[]

  /**
   * Returns the cached fork-scoped Layer (built by initFork). Includes
   * WorkingDirectory, all reader
   * services, ToolInterceptor, etc. Used by Cortex to provide tool-execution
   * context for the native paradigm.
   */
  readonly getForkLayer: (forkId: string | null) => ForkLayer | undefined

  readonly fork: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message: string
    outputSchema?: JsonSchema | undefined
    mode: 'clone' | 'spawn'
    role: RoleId
    taskId: string
  }) => Effect.Effect<
    string,
    never,
    ExecutionProjectionRequirements | ChatPersistence | WorkerBusService<AppEvent>
  >

}

export class ExecutionManager extends Context.Tag('ExecutionManager')<
  ExecutionManager,
  ExecutionManagerService
>() {}
