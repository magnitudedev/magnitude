/**
 * ExecutionManager
 *
 * Owns per-fork lifecycle and turn execution.
 * Consumes TurnEngineEvent stream, applies policy, emits TurnEvents.
 */

import * as path from 'path'
import { Effect, Stream, Layer, Ref } from 'effect'
import {
  ToolInterceptorTag,
  type TurnEngineEvent,
  type ToolInterceptor,
} from '@magnitudedev/turn-engine'
import { Fork, Projection, WorkerBusTag, AmbientServiceTag, type WorkerBusService, type AmbientService } from '@magnitudedev/event-core'
import { SkillsAmbient } from '../ambient/skills-ambient'
import type {
  AppEvent,
  ToolResult,
  MessageDestination,
  TurnFeedback,
  TurnCompletion,
  TurnOutcome,
} from '../events'
import { catalog, isToolKey, type ToolKey } from '../catalog'
import { buildRegisteredTools } from '../tools/tool-registry'

import { isValidVariant, type AgentVariant } from '../agents/variants'
import { getAgentDefinition, getAgentSlot } from '../agents/registry'
import { buildPolicyInterceptor, type AgentResolver } from './permission-gate'
export { IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from './types'
import { createApprovalState, ApprovalStateTag, type ApprovalStateService } from './approval-state'

import { BrowserService } from '../services/browser-service'
import { BrowserHarnessTag } from '../tools/browser-tools'

import { AgentStateReaderTag, type AgentStateReader } from '../tools/fork'
import { AgentRegistryStateReaderTag, type AgentRegistryStateReader } from '../tools/agent-registry-reader'
import { buildCloneContext, buildSpawnContext } from '../prompts/fork-context'
import { formatTaskOutsideSubtreeError } from '../prompts/error-states'
import type { JsonSchema } from '@magnitudedev/llm-core'
import { ConversationStateReaderTag, type ConversationStateReader } from '../tools/memory-reader'
import { TaskGraphStateReaderTag, canCompleteRecord, getChildRecords, canAssignRecord, collectSubtreeRecords } from '../tools/task-reader'
import { ConversationProjection, type ConversationState } from '../projections/conversation'
import { createId } from '../util/id'
import { logger } from '@magnitudedev/logger'

import { AgentRoutingProjection, type AgentRoutingState, getRoutingEntryByForkId } from '../projections/agent-routing'
import { AgentStatusProjection, type AgentStatusState, getActiveAgent, getAgentByForkId } from '../projections/agent-status'
import { TurnProjection, type ForkTurnState } from '../projections/turn'
import { SessionContextProjection, type SessionContextState } from '../projections/session-context'
// ReplayProjection kept for backward compat but not used in execute()
import { TaskGraphProjection, type TaskGraphState, type TaskStatus } from '../projections/task-graph'

import type { RoleDefinition, BoundObservable } from '@magnitudedev/roles'
import { bindObservable } from '@magnitudedev/roles'
import { ProjectionReaderTag, type ProjectionReader } from '../observables/projection-reader'
import { EphemeralSessionContextTag, PolicyContextProviderTag, type EphemeralSessionContext, type PolicyContext } from '../agents/types'
import { createPolicyContextProvider } from '../agents/policy-context'
import { ExecutionManager, IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from './types'
import type { TurnEvent, TurnEventSink, ExecuteOptions, ExecuteResult, ExecutionManagerService } from './types'
import { WorkingDirectoryTag } from './working-directory'
import type { StreamingLeaf, StreamHook, ToolContext, ToolDefinition } from '@magnitudedev/tools'


import { ChatPersistence } from '../persistence/chat-persistence-service'

const { ForkContext } = Fork

type AgentDef = RoleDefinition

type AnyStreamHook = StreamHook<Record<string, unknown>, unknown, unknown, unknown, unknown>
type StreamableTool = ToolDefinition & { stream: AnyStreamHook }

function hasStreamHook<T extends ToolDefinition>(tool: T): tool is T & StreamableTool {
  return 'stream' in tool && !!tool.stream
}

import { mapXmlToolResult } from '../util/tool-result'
import { handleTaskDirective } from '../tasks/operations'
import { persistResult } from '../runtime/result-persistence'


// =============================================================================
// Types
// =============================================================================



// =============================================================================
// Implementation
// =============================================================================

/**
 * Build the unified Effect layer for a fork — covers tool execution, interceptor, and emit.
 * Tools use reader services, interceptor uses PolicyContextProvider + ApprovalState.
 */
function makeForkLayers(
  forkId: string | null,
  slot: string,

  sessionContextProjection: Projection.ProjectionInstance<SessionContextState>,
  agentProjection: Projection.ProjectionInstance<AgentRoutingState>,
  agentStatusProjection: Projection.ProjectionInstance<AgentStatusState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkTurnState>,
  taskGraphProjection: Projection.ProjectionInstance<TaskGraphState>,

  conversationProjection: Projection.ProjectionInstance<ConversationState>,
  approvalState: ApprovalStateService,
  persistenceLayer: Layer.Layer<ChatPersistence, never, never>,
  policyInterceptor: ReturnType<typeof buildPolicyInterceptor>,

  cwd: string,
  workspacePath: string,
  ephemeralSessionContext: EphemeralSessionContext,
) {
  const agentRegistryStateReaderLayer = Layer.succeed(AgentRegistryStateReaderTag, {
    getState: () => agentStatusProjection.get
  } satisfies AgentRegistryStateReader)

  const conversationStateReaderLayer = Layer.succeed(ConversationStateReaderTag, {
    getState: () => conversationProjection.get
  } satisfies ConversationStateReader)

  const agentStateReaderLayer = Layer.succeed(AgentStateReaderTag, {
    getAgentState: () => agentStatusProjection.get,
    getAgent: (agentId: string) => Effect.map(agentStatusProjection.get, (state) => state.agents.get(agentId)),
  } satisfies AgentStateReader)

  const taskGraphReaderLayer = Layer.succeed(TaskGraphStateReaderTag, {
    getTask: (id) => Effect.map(taskGraphProjection.get, (s) => s.tasks.get(id)),
    getState: () => taskGraphProjection.get,
    getChildren: (id) => Effect.map(taskGraphProjection.get, (s) => getChildRecords(s, id)),
    canComplete: (id) => Effect.map(taskGraphProjection.get, (s) => canCompleteRecord(s, id)),
    canAssign: (id, assignee) => Effect.map(taskGraphProjection.get, (s) => canAssignRecord(s, id, assignee)),
    getSubtree: (id) => Effect.map(taskGraphProjection.get, (s) => collectSubtreeRecords(s, id)),
  })

  const policyCtxProvider = createPolicyContextProvider(
    forkId,
    cwd,
    workspacePath,
    ephemeralSessionContext,
    agentStatusProjection,
    workingStateProjection,
  )

  const providedInterceptor: ToolInterceptor = {
    beforeExecute: (ctx) =>
      policyInterceptor(ctx).pipe(
        Effect.provideService(ForkContext, { forkId, slot }),
        Effect.provideService(PolicyContextProviderTag, policyCtxProvider),
        Effect.provideService(ApprovalStateTag, approvalState),
      ),
  }

  return Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId, slot }),

    agentRegistryStateReaderLayer,
    conversationStateReaderLayer,
    taskGraphReaderLayer,
    agentStateReaderLayer,


    Layer.succeed(ApprovalStateTag, approvalState),
    Layer.succeed(WorkingDirectoryTag, { cwd, workspacePath }),
    Layer.succeed(EphemeralSessionContextTag, ephemeralSessionContext),
    Layer.succeed(PolicyContextProviderTag, policyCtxProvider),
    Layer.succeed(ToolInterceptorTag, providedInterceptor),
    persistenceLayer,
  )
}

/**
 * Create the execution manager.
 * No sandboxes — event stream is consumed directly per execute() call.
 */
const makeExecutionManager = Effect.gen(function* () {
  const ephemeralSessionContext = yield* EphemeralSessionContextTag
  // Per-fork cached layers (built during initFork, reused across turns)
  const forkLayers = new Map<string | null, Layer.Layer<never>>()
  const forkCwds = new Map<string | null, string>()
  const forkWorkspacePaths = new Map<string | null, string | undefined>()




  // Bound observables map
  const boundObservables = new Map<string | null, BoundObservable[]>()

  // Session-level oneshot mode flag (set once during root fork init, never changes)
  let oneshotEnabled = false

  // Approval state for gated tool calls
  const approvalState = createApprovalState()
  // Maps forkId → variant, populated when forks are created.
  const forkAgentVariants = new Map<string, AgentVariant>()

  // Pre-built teardown effects (captured at initFork time with services already provided)
  const forkTeardowns = new Map<string, Effect.Effect<void>>()

  // Pre-built idle release effects (repeatable; run each time fork goes idle)
  const forkIdleReleases = new Map<string, Effect.Effect<void>>()

  // Per-fork consecutive identical continue-response tracker
  const identicalContinueTracker = new Map<string | null, { lastResponseText: string; consecutiveCount: number }>()



  /**
   * Resolve the active agent definition for a fork.
   * Child forks use their fixed role. Root fork uses the orchestrator definition.
   */
  const resolveAgent: AgentResolver = (forkId) => {
    if (forkId !== null) {
      const variant = forkAgentVariants.get(forkId) ?? 'worker'
      return getAgentDefinition(variant)
    }
    return getAgentDefinition('lead')
  }

  // Build the policy interceptor (shared across all forks, resolves agent dynamically)
  const policyInterceptor = buildPolicyInterceptor(resolveAgent)

  function buildForkContext(params: { mode: string; prompt: string; outputSchema?: JsonSchema | undefined }) {
    return Effect.gen(function* () {
      if (params.mode === 'clone') {
        return buildCloneContext(params.prompt, params.outputSchema)
      }
      const proj = yield* SessionContextProjection.Tag
      const ctx = yield* Effect.map(proj.get, s => s.context)
      return buildSpawnContext(params.prompt, ctx, params.outputSchema)
    })
  }

  const service: ExecutionManagerService = {
    execute: (eventStream, options, sink) => Effect.gen(function* () {
      const { forkId, turnId, defaultProseDest, triggeredByUser } = options

      const ambientService = yield* AmbientServiceTag
      const skills = ambientService.getValue(SkillsAmbient)

      // Resolve agent definition for this fork
      const agentRoutingProjectionInst = yield* AgentRoutingProjection.Tag
      const agentStatusProjectionInst = yield* AgentStatusProjection.Tag
      const workingStateProjectionInst = yield* TurnProjection.Tag
      const agentState = yield* agentStatusProjectionInst.get
      let variant: AgentVariant
      if (forkId) {
        const agentInstance = getAgentByForkId(agentState, forkId)
        const role = agentInstance?.role
        variant = role && isValidVariant(role) ? role : 'worker'
      } else {
        variant = 'lead'
      }
      const agentDef = getAgentDefinition(variant)

      // Get cached fork layers (must be initialized via initFork)
      const layers = forkLayers.get(forkId)
      if (!layers) {
        return yield* Effect.die(
          new Error(`Fork not initialized: ${forkId}. initFork() must be called before execute().`)
        )
      }

      const executionLayer = layers

      const workspacePath = forkWorkspacePaths.get(forkId)!

      // Build registered tools for tool key resolution and stream hooks
      const registeredTools = buildRegisteredTools(options.toolSet, executionLayer)
      const resultsDir = path.join(workspacePath, 'results')

      /** Resolve a tool's model-facing name to the internal catalog key. */
      const resolveKey = (toolName: string): ToolKey | undefined => {
        const rt = registeredTools.get(toolName)
        if (!rt) return undefined
        const meta = rt.meta as { defKey?: unknown } | undefined
        const defKey = typeof meta?.defKey === 'string' ? meta.defKey : undefined
        return defKey && isToolKey(defKey) ? defKey as ToolKey : undefined
      }

      // Track tools called (by definition key) for turn policy
      const toolsCalledKeys: ToolKey[] = []
      let lastToolKey: ToolKey | null = null
      let hasToolErrors = false
      const messagesSent: Array<{ id: string, taskId: string | null }> = []
      let hasAnyMessage = false
      let hasAnyResponseContent = false
      let directUserRepliesSent = 0

      const isTaskInAssignedSubtree = (
        taskState: TaskGraphState,
        candidateParentId: string,
        assignedTaskId: string,
      ): boolean => {
        let current: string | null = candidateParentId
        while (current !== null) {
          if (current === assignedTaskId) return true
          current = taskState.tasks.get(current)?.parentId ?? null
        }
        return false
      }

      // Track tool input (ToolInputReady provides the parsed input)
      const toolInputs = new Map<string, unknown>()

      // Content fingerprint for circuit breaker (replaces raw XML text capture)
      let contentFingerprint = ''

      // Streaming hook state per tool call
      const streamHookStates = new Map<string, unknown>()
      interface StreamHookEntry {
        readonly hook: AnyStreamHook
        readonly layerProvider?: () => Effect.Effect<Layer.Layer<never>, unknown>
      }

      const streamHookConfigs = new Map<string, StreamHookEntry>()
      // Simple field accumulator for streaming partial input
      const streamingFields = new Map<string, Record<string, StreamingLeaf<unknown>>>()

      /** Invoke a tool's stream hook with current streaming state.
       *  Uses the registered tool's layerProvider to satisfy R requirements internally,
       *  Uses the registered tool's layerProvider to satisfy R requirements internally. */
      const invokeStreamHook = (toolCallId: string, toolKey: ToolKey): Effect.Effect<void> =>
        Effect.suspend(() => {
          const entry = streamHookConfigs.get(toolCallId)
          if (!entry) return Effect.void

          const currentState = streamHookStates.get(toolCallId)
          const partialInput = streamingFields.get(toolCallId) ?? {}
          const { hook, layerProvider } = entry

          const streamCtx: ToolContext<unknown> = {
            emit: (value) => sink.emit({
              _tag: 'ToolEvent' as const,
              toolCallId,
              toolKey,
              event: { _tag: 'ToolEmission' as const, toolCallId, value },
            }),
          }

          const effect = hook.onInput(partialInput, currentState, streamCtx).pipe(
            Effect.tapError((e) => Effect.sync(() => {
              logger.error(`[ExecutionManager] Stream hook error for ${toolKey} (${toolCallId}): ${e}`)
            })),
            Effect.catchAll(() => Effect.succeed(currentState)),
          )

          const provided = layerProvider
            ? layerProvider().pipe(Effect.flatMap((layer) => effect.pipe(Effect.provide(layer))))
            : effect

          return provided.pipe(
            Effect.map((newState) => { streamHookStates.set(toolCallId, newState) }),
          ) as Effect.Effect<void>
        })

      // toolCallId → ToolKey tracking
      const toolCallKeys = new Map<string, ToolKey>()

      const feedback: TurnFeedback[] = []
      let sawToolExecutionError = false

      // Store execution result
      let executionResult: TurnOutcome = { _tag: 'Completed', completion: { toolCallsCount: 0, finishReason: 'stop', feedback: [] } }
      let turnUsage: { inputTokens: number | null; outputTokens: number | null; cacheReadTokens: number | null; cacheWriteTokens: number | null } | null = null

      // Create the PolicyContextProvider for turn policy evaluation
      const cwd = forkCwds.get(forkId) ?? process.cwd()

      const policyCtxProvider = createPolicyContextProvider(
        forkId,
        cwd,
        workspacePath,
        ephemeralSessionContext,
        yield* AgentStatusProjection.Tag,
        yield* TurnProjection.Tag,
      )


      yield* Effect.scoped(
        eventStream.pipe(
          Stream.runForEach((event: TurnEngineEvent) => Effect.gen(function* () {
            switch (event._tag) {
              // --- Tool Input Started ---
              case 'ToolInputStarted': {
                hasAnyResponseContent = true
                const toolKey = resolveKey(event.toolName)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Failed to resolve tool key for ${event.toolName} (toolCallId: ${event.toolCallId}).`)
                  break
                }
                toolCallKeys.set(event.toolCallId, toolKey)

                // Check for stream hook
                const toolEntry = catalog.entries[toolKey]
                const tool = toolEntry?.tool
                if (tool && hasStreamHook(tool)) {
                  const streamConfig = tool.stream
                  const registered = registeredTools.get(event.toolName)
                  streamHookStates.set(event.toolCallId, streamConfig.initial)
                  streamHookConfigs.set(event.toolCallId, { hook: streamConfig, layerProvider: registered?.layerProvider })
                  streamingFields.set(event.toolCallId, {})
                }

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Input Ready (cache the parsed input) ---
              case 'ToolInputReady': {
                toolInputs.set(event.toolCallId, event.input)

                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }

                const fields = streamingFields.get(event.toolCallId)
                if (fields) {
                  for (const [key, field] of Object.entries(fields)) {
                    if (!field.isFinal) fields[key] = { value: field.value, isFinal: true }
                  }
                }

                yield* invokeStreamHook(event.toolCallId, toolKey)

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Input Decode Failure ---
              case 'ToolInputDecodeFailure': {
                hasAnyResponseContent = true
                const toolKey = resolveKey(event.toolName)
                if (!toolKey) break
                toolCallKeys.set(event.toolCallId, toolKey)

                toolsCalledKeys.push(toolKey)
                lastToolKey = toolKey

                hasToolErrors = true

                yield* sink.emit({
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Turn Structure Decode Failure ---
              case 'TurnStructureDecodeFailure': {
                hasAnyResponseContent = true
                break
              }

              // --- Tool Execution Started ---
              case 'ToolExecutionStarted': {
                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }
                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              case 'ToolEmission': {
                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }
                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Execution Ended ---
              case 'ToolExecutionEnded': {
                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }

                // Track tool calls for turn policy
                toolsCalledKeys.push(toolKey)
                lastToolKey = toolKey

                toolInputs.delete(event.toolCallId)
                streamHookStates.delete(event.toolCallId)
                streamHookConfigs.delete(event.toolCallId)
                streamingFields.delete(event.toolCallId)

                const toolResult: ToolResult = mapXmlToolResult(event.result)
                if (toolResult.status === 'error') {
                  hasToolErrors = true
                  sawToolExecutionError = true
                }

                // Persist successful tool results
                if (event.result._tag === 'Success') {
                  yield* persistResult(event.result.output, turnId, event.toolCallId, resultsDir).pipe(
                    Effect.catchAll((e) => Effect.gen(function* () {
                      logger.warn({ forkId, turnId, toolCallId: event.toolCallId, e }, '[ExecutionManager] persistResult failed')
                    })),
                  )
                }

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool input field chunk events ---
              case 'ToolInputFieldChunk': {
                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }

                // Update streaming fields (accumulate raw text per field)
                const fields = streamingFields.get(event.toolCallId)
                if (fields) {
                  const existing = fields[event.field]
                  const prev = existing && !existing.isFinal ? String(existing.value) : ''
                  fields[event.field] = { value: prev + event.delta, isFinal: false }
                }

                // Invoke stream hook if present
                yield* invokeStreamHook(event.toolCallId, toolKey)

                // Accumulate fingerprint for circuit breaker
                contentFingerprint += event.delta

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              case 'ToolInputFieldComplete': {
                const toolKey = toolCallKeys.get(event.toolCallId)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Tool key not found for toolCallId ${event.toolCallId} (event: ${event._tag}).`)
                  break
                }

                // Mark the completed field as final in streamingFields
                const fields = streamingFields.get(event.toolCallId)
                if (fields) {
                  const existing = fields[event.field]
                  if (existing) {
                    fields[event.field] = { value: existing.value, isFinal: true }
                  }
                }

                // Invoke stream hook if present — the completed field is now isFinal=true
                yield* invokeStreamHook(event.toolCallId, toolKey)

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Messages / Think prose ---
              case 'MessageStart': {
                hasAnyResponseContent = true
                const taskProjection = yield* TaskGraphProjection.Tag
                const taskState = yield* taskProjection.get

                const explicitTo = typeof event.to === 'string' ? event.to : null

                // Resolve destination
                let destination: MessageDestination

                if (explicitTo !== null) {
                  if (explicitTo === 'user') {
                    if (forkId !== null) {
                      feedback.push({
                        _tag: 'InvalidMessageDestination',
                        destination: explicitTo,
                        message: `Invalid message destination "${explicitTo}": only root fork can send to user`,
                      })
                      break
                    }
                    destination = { kind: 'user' }
                  } else if (explicitTo === 'parent') {
                    if (forkId === null) {
                      feedback.push({
                        _tag: 'InvalidMessageDestination',
                        destination: explicitTo,
                        message: `Invalid message destination "${explicitTo}": root fork has no parent`,
                      })
                      break
                    }
                    destination = { kind: 'parent' }
                  } else {
                    const targetTask = taskState.tasks.get(explicitTo)
                    if (!targetTask) {
                      feedback.push({
                        _tag: 'InvalidMessageDestination',
                        destination: explicitTo,
                        message: `Invalid message destination "${explicitTo}": task not found`,
                      })
                      break
                    }
                    if (!targetTask.worker) {
                      feedback.push({
                        _tag: 'InvalidMessageDestination',
                        destination: explicitTo,
                        message: `Invalid message destination "${explicitTo}": task has no active worker`,
                      })
                      break
                    }
                    destination = { kind: 'worker', taskId: explicitTo }
                  }
                } else {
                  // No explicit `to` — resolve from context
                  if (forkId === null) {
                    destination = { kind: 'user' }
                  } else {
                    destination = defaultProseDest === 'user' ? { kind: 'user' } : { kind: 'parent' }
                  }
                }

                // Skip message directive for worker-targeted messages —
                // they don't affect user reply counting
                if (destination.kind !== 'worker') {
                  const messageResult = yield* handleTaskDirective({
                    kind: 'message',
                    defaultTopLevelDestination: defaultProseDest,
                    triggeredByUser,
                    directUserRepliesSent,
                  }, { forkId, timestamp: Date.now(), graph: { tasks: new Map() }, skills }).pipe(
                    Effect.provideService(ForkContext, { forkId, slot: options.toolSet.slot }),
                    Effect.provide(executionLayer),
                  )

                  if (!messageResult.success) {
                    break
                  }

                  if (
                    'directUserRepliesSent' in messageResult
                    && typeof messageResult.directUserRepliesSent === 'number'
                  ) {
                    directUserRepliesSent = messageResult.directUserRepliesSent
                  }
                }

                messagesSent.push({ id: event.id, taskId: destination.kind === 'worker' ? destination.taskId : null })
                hasAnyMessage = true

                yield* sink.emit( {
                  _tag: 'MessageStart',
                  id: event.id,
                  destination,
                })
                break
              }

              case 'MessageChunk': {
                contentFingerprint += event.text
                yield* sink.emit( { _tag: 'MessageChunk', id: event.id, text: event.text })
                break
              }

              case 'MessageEnd': {
                yield* sink.emit( { _tag: 'MessageEnd', id: event.id })
                break
              }

              // --- Thinking ---
              case 'ThoughtStart': {
                hasAnyResponseContent = true
                yield* sink.emit({ _tag: 'ThinkingStart' })
                break
              }

              case 'ThoughtChunk': {
                hasAnyResponseContent = true
                contentFingerprint += event.text
                yield* sink.emit({ _tag: 'ThinkingDelta', text: event.text })
                break
              }

              case 'ThoughtEnd': {
                hasAnyResponseContent = true
                yield* sink.emit({ _tag: 'ThinkingEnd' })
                break
              }


              // --- Turn End ---
              case 'TurnEnd': {
                const outcome = event.outcome
                // Capture usage for return
                if (event.usage) {
                  turnUsage = {
                    inputTokens: event.usage.inputTokens ?? null,
                    outputTokens: event.usage.outputTokens ?? null,
                    cacheReadTokens: event.usage.cacheReadTokens ?? null,
                    cacheWriteTokens: event.usage.cacheWriteTokens ?? null,
                  }
                }

                const completed = (toolCallsCount: number): TurnOutcome => ({
                  _tag: 'Completed',
                  completion: {
                    toolCallsCount,
                    finishReason: toolCallsCount > 0 ? 'tool_calls' : 'stop',
                    feedback: [...feedback],
                  } satisfies TurnCompletion,
                })

                switch (outcome._tag) {
                  case 'Completed': {
                    let willContinue: boolean

                    if (hasToolErrors || feedback.length > 0) {
                      // Errors or feedback → retrigger so model sees them
                      willContinue = true
                    } else if (!hasAnyResponseContent) {
                      // Empty response → retrigger so corrective feedback is visible
                      willContinue = true
                    } else {
                      // Use agent policy to decide
                      const policyCtx = yield* policyCtxProvider.get
                      const turnResult = agentDef.getTurn({
                        toolsCalled: toolsCalledKeys,
                        lastTool: lastToolKey,
                        messagesSent,
                        state: policyCtx,
                      })
                      willContinue = turnResult.action === 'continue'
                    }

                    // Oneshot liveness guard: prevent stalling when nothing is active
                    if (!willContinue && oneshotEnabled) {
                      const pCtx = yield* policyCtxProvider.get
                      if (pCtx.activeAgentCount === 0) {
                        feedback.push({ _tag: 'OneshotLivenessRetriggered' })
                        willContinue = true
                      }
                    }

                    executionResult = completed(willContinue ? Math.max(outcome.toolCallsCount, 1) : 0)
                    break
                  }

                  case 'ToolInputDecodeFailure':
                    executionResult = { _tag: 'ParseFailure', error: {
                      _tag: 'ToolInputDecodeFailure' as const,
                      toolCallId: outcome.toolCallId,
                      toolName: outcome.toolName,
                      group: '',
                      detail: outcome.detail,
                    }}
                    break

                  case 'TurnStructureDecodeFailure':
                    executionResult = { _tag: 'ParseFailure', error: {
                      _tag: 'TurnStructureDecodeFailure' as const,
                      detail: outcome.detail,
                    }}
                    break

                  case 'GateRejected':
                    executionResult = completed(1)
                    break

                  case 'EngineDefect':
                    executionResult = {
                      _tag: 'UnexpectedError',
                      message: outcome.message,
                      detail: { _tag: 'EngineDefect' },
                    }
                    break
                }

                // Circuit breaker: stop tight loops of identical consecutive responses that would retrigger.
                const willRetrigger =
                  (executionResult._tag === 'Completed' && executionResult.completion.toolCallsCount > 0)
                  || executionResult._tag === 'ParseFailure'

                if (willRetrigger) {
                  const previous = identicalContinueTracker.get(forkId)
                  const nextCount = previous && previous.lastResponseText === contentFingerprint
                    ? previous.consecutiveCount + 1
                    : 1

                  identicalContinueTracker.set(forkId, {
                    lastResponseText: contentFingerprint,
                    consecutiveCount: nextCount,
                  })

                  if (nextCount >= IDENTICAL_RESPONSE_BREAKER_THRESHOLD) {
                    executionResult = {
                      _tag: 'SafetyStop',
                      reason: {
                        _tag: 'IdenticalResponseCircuitBreaker',
                        threshold: nextCount,
                      },
                    }
                    identicalContinueTracker.delete(forkId)
                  }
                } else {
                  identicalContinueTracker.delete(forkId)
                }
                break
              }
            }
          }))
        )
      )

      return {
        result: executionResult,
        usage: turnUsage,
      }
    }),

    initFork: (forkId, variant) => (Effect.gen(function* () {
      yield* WorkerBusTag<AppEvent>()

      const sessionContextProjection = yield* SessionContextProjection.Tag
      const agentProjection = yield* AgentRoutingProjection.Tag
      const agentStatusProjection = yield* AgentStatusProjection.Tag
      const workingStateProjection = yield* TurnProjection.Tag
      const taskGraphProjection = yield* TaskGraphProjection.Tag

      const conversationProjection = yield* ConversationProjection.Tag
      const persistence = yield* ChatPersistence
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)

      const sessionState = yield* sessionContextProjection.get
      if (!sessionState.context) {
        return yield* Effect.die(
          new Error('Session context not initialized. session_initialized must be processed before initFork().'),
        )
      }
      const cwd = sessionState.context.cwd
      const workspacePath = sessionState.context.workspacePath
      if (forkId === null) {
        oneshotEnabled = !!sessionState.context?.oneshot
      }

      const slot = getAgentSlot(variant)
      
      let layers = makeForkLayers(
        forkId,
        slot,
        sessionContextProjection, agentProjection, agentStatusProjection,
        workingStateProjection, taskGraphProjection,
        conversationProjection,
        approvalState,
        persistenceLayer, policyInterceptor, cwd, workspacePath, ephemeralSessionContext,
      )
      forkCwds.set(forkId, cwd)
      forkWorkspacePaths.set(forkId, workspacePath)

      // Inject role-specific setup layer when the role defines a setup function
      const roleDef = getAgentDefinition(variant)
      if (roleDef.setup && forkId) {
        const setupLayer = (yield* roleDef.setup({ forkId, cwd, workspacePath })) as Layer.Layer<never>
        layers = Layer.merge(layers, setupLayer)
      }

      // Pre-build teardown effect with services captured now (so disposeFork needs no requirements)
      if (forkId && (roleDef.setup || roleDef.teardown)) {
        const browserService = yield* BrowserService

        if (roleDef.teardown) {
          const teardownEffect = roleDef.teardown({ forkId, cwd, workspacePath }).pipe(
            Effect.provideService(BrowserService, browserService)
          ) as Effect.Effect<void>
          forkTeardowns.set(forkId, teardownEffect)
        }

        // Store repeatable idle release (for browser forks)
        forkIdleReleases.set(forkId, browserService.release(forkId) as Effect.Effect<void>)
      }

      // Store variant for agent resolution
      if (forkId !== null) {
        forkAgentVariants.set(forkId, variant)
      }

      const projectionReader: ProjectionReader = {
        getAgentRouting: () => agentProjection.get,
        getAgentStatus: () => agentStatusProjection.get,
      }
      const projectionReaderLayer = Layer.succeed(ProjectionReaderTag, projectionReader)
      layers = Layer.merge(layers, projectionReaderLayer)

      // Cache the layers
      forkLayers.set(forkId, layers)

      // Bind observables
      const agentDef = getAgentDefinition(variant)
      const agentObservables = agentDef.observables.map((obs) =>
        bindObservable(obs, () => Effect.succeed(layers as Layer.Layer<unknown>))
      )
      boundObservables.set(forkId, agentObservables)
    }) as Effect.Effect<void, never, Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ForkTurnState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>>),

    disposeFork: (forkId) => Effect.gen(function* () {
      // Run role teardown if defined (e.g. browser cleanup)
      const teardown = forkTeardowns.get(forkId)
      if (teardown) {
        yield* Effect.ignore(teardown)
        forkTeardowns.delete(forkId)
      }

      forkLayers.delete(forkId)
      forkCwds.delete(forkId)
      forkWorkspacePaths.delete(forkId)

      boundObservables.delete(forkId)
      forkAgentVariants.delete(forkId)
      forkIdleReleases.delete(forkId)
      identicalContinueTracker.delete(forkId)
    }),

    fork: (params: {
      parentForkId: string | null
      name: string
      agentId: string
      prompt: string
      message: string
      outputSchema?: JsonSchema | undefined
      mode: 'clone' | 'spawn'
      role: AgentVariant
      taskId: string
    }) => Effect.gen(function* () {
      const forkId = createId()
      forkAgentVariants.set(forkId, params.role)
      const workerBus = yield* WorkerBusTag<AppEvent>()
      const context = yield* buildForkContext(params)

      yield* service.initFork(forkId, params.role)

      const taskId = params.taskId.trim()
      if (taskId.length === 0) {
        return yield* Effect.die(new Error('ExecutionManager.fork requires a non-empty taskId'))
      }

      yield* workerBus.publish({
        type: 'agent_created',
        forkId,
        parentForkId: params.parentForkId,
        agentId: params.agentId,
        name: params.name,
        role: params.role,
        context,
        mode: params.mode,
        taskId,
        message: params.message,
        outputSchema: params.outputSchema,
      })

      return forkId
    }),

    releaseBrowserFork: (forkId) => Effect.gen(function* () {
      const release = forkIdleReleases.get(forkId)
      if (release) {
        yield* Effect.ignore(release)
      }
    }),

    approvalState,


    getObservables: (forkId) => boundObservables.get(forkId) ?? [],

    getForkLayer: (forkId) => forkLayers.get(forkId),
  }

  return service
})


// =============================================================================
// Layer
// =============================================================================

/**
 * ExecutionManager layer - no external requirements.
 * Services (projections) are accessed lazily at execution time.
 */
export const ExecutionManagerLive = Layer.scoped(
  ExecutionManager,
  makeExecutionManager
)