/**
 * Execution Types
 *
 * Types for the turn event stream, execution manager service interface,
 * and the ExecutionManager Effect service tag.
 *
 * This is a leaf module — it uses only type imports from the rest of the agent
 * package, breaking the circular dependency:
 *   catalog → models → tools → execution-manager → catalog
 */

import { Effect, Data, Context, Stream } from 'effect'
import type { TurnEngineEvent, TurnEngineCrash, EngineState } from '@magnitudedev/xml-act'
import type { MessageDestination, TurnResult } from '../events'
import type { CallUsage, ModelError } from '@magnitudedev/providers'
import type { Projection, WorkerBusService, AmbientService } from '@magnitudedev/event-core'
import type { AgentVariant } from '../agents/variants'
import type { AgentRoutingState } from '../projections/agent-routing'
import type { AgentStatusState } from '../projections/agent-status'
import type { TaskGraphState } from '../projections/task-graph'
import type { ForkTurnState } from '../projections/turn'
import type { SessionContextState } from '../projections/session-context'
import type { ConversationState } from '../projections/conversation'
import type { TaskGraphStateReaderTag } from '../tools/task-reader'
import type { ConversationStateReaderTag } from '../tools/memory-reader'
import type { ApprovalStateService } from './approval-state'
import type { BrowserService } from '../services/browser-service'
import type { ChatPersistence } from '../persistence/chat-persistence-service'
import type { BoundObservable } from '@magnitudedev/roles'
import type { JsonSchema } from '@magnitudedev/llm-core'
import type { AppEvent } from '../events'
import type { ResolvedToolSet } from '../tools/resolved-toolset'


// =============================================================================
// Turn Events
// =============================================================================

/**
 * Events yielded during turn execution.
 * Cortex maps these to AppEvents and publishes them to the event bus.
 *
 * These carry only the data the execution manager naturally has.
 * Cortex decorates with forkId, turnId, chainId when publishing.
 */
export type TurnEvent =
  // --- Message/thinking content ---
  | { readonly _tag: 'MessageStart'; readonly id: string; readonly destination: MessageDestination }
  | { readonly _tag: 'MessageChunk'; readonly id: string; readonly text: string }
  | { readonly _tag: 'MessageEnd'; readonly id: string }
  | { readonly _tag: 'ThinkingDelta'; readonly text: string }
  | { readonly _tag: 'ThinkingEnd'; readonly about: string | null }
  | { readonly _tag: 'RawResponseChunk'; readonly text: string }
  | { readonly _tag: 'LensStarted'; readonly name: string }
  | { readonly _tag: 'LensDelta'; readonly text: string }
  | { readonly _tag: 'LensEnded'; readonly name: string }

  // --- Tool events (forwarded xml-act TurnEngineEvent with agent metadata) ---
  | { readonly _tag: 'ToolEvent'; readonly toolCallId: string; readonly toolKey: string; readonly event: TurnEngineEvent }

  // --- Terminal (always last event in the stream) ---
  | { readonly _tag: 'TurnResult'; readonly value: TurnStrategyResult }

export interface TurnEventSink {
  readonly emit: (event: TurnEvent) => Effect.Effect<void>
}

// =============================================================================
// Turn Error
// =============================================================================

/**
 * Typed errors from turn execution.
 * Auth failures, LLM API errors, and stream read errors.
 */
export type TurnError = Data.TaggedEnum<{
  /** OAuth / API key validation failure */
  readonly AuthFailed: { readonly message: string; readonly cause?: unknown }
  /** LLM API returned an error (HTTP error, validation error, etc.) */
  readonly LLMFailed: { readonly message: string; readonly cause?: unknown }
  /** Error reading from the LLM response stream */
  readonly StreamFailed: { readonly message: string; readonly cause?: unknown }
}>

export const TurnError = Data.taggedEnum<TurnError>()

// =============================================================================
// Turn Result
// =============================================================================

export interface TurnStrategyResult {
  readonly executeResult: ExecuteResult
  readonly usage: CallUsage
}

// =============================================================================
// ExecutionManager Service
// =============================================================================

export const IDENTICAL_RESPONSE_BREAKER_THRESHOLD = 5

export interface ExecuteOptions {
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly defaultProseDest: 'user' | 'parent'
  readonly triggeredByUser: boolean
  readonly toolSet: ResolvedToolSet
}

export interface ExecuteResult {
  readonly result: TurnResult
}

export interface ExecutionManagerService {
  readonly execute: (
    xmlStream: Stream.Stream<string, ModelError>,
    options: ExecuteOptions,
    sink: TurnEventSink,
  ) => Effect.Effect<
    ExecuteResult,
    TurnEngineCrash,
    Projection.ProjectionInstance<AgentRoutingState>
    | Projection.ProjectionInstance<AgentStatusState>
    | Projection.ProjectionInstance<TaskGraphState>
    | Projection.ForkedProjectionInstance<EngineState>
    | Projection.ForkedProjectionInstance<ForkTurnState>
    | WorkerBusService<AppEvent>
    | TaskGraphStateReaderTag
    | ConversationStateReaderTag
    | AmbientService
  >

  readonly initFork: (
    forkId: string | null,
    variant: AgentVariant
  ) => Effect.Effect<
    void,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ForkTurnState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

  readonly disposeFork: (forkId: string) => Effect.Effect<void>

  readonly getObservables: (forkId: string | null) => BoundObservable[]

  readonly fork: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message: string
    outputSchema?: JsonSchema | undefined
    mode: 'clone' | 'spawn'
    role: AgentVariant
    taskId: string
  }) => Effect.Effect<
    string,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ForkTurnState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

  readonly releaseBrowserFork: (forkId: string) => Effect.Effect<void>

  readonly approvalState: ApprovalStateService
}

export class ExecutionManager extends Context.Tag('ExecutionManager')<
  ExecutionManager,
  ExecutionManagerService
>() {}
