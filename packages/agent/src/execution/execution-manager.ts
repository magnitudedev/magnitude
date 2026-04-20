/**
 * ExecutionManager
 *
 * Owns per-fork lifecycle and xml-act runtime execution.
 * Maps TurnEngineEvents to agent TurnEvents.
 *
 * No sandboxes, no journals, no WASM — just xml-act streaming runtime.
 */

import * as path from 'path'
import { Effect, Stream, Layer, Ref } from 'effect'
import type { ModelError } from '@magnitudedev/providers'
import {
  createTurnEngine,
  ToolInterceptorTag,
  TurnEngineCrash,
  type TurnEngineEvent,
  type ToolInterceptor,
  type EngineState,
} from '@magnitudedev/xml-act'
import { Fork, Projection, WorkerBusTag, AmbientServiceTag, type WorkerBusService, type AmbientService } from '@magnitudedev/event-core'
import { SkillsAmbient } from '../ambient/skills-ambient'
import type { AppEvent, TurnResult, TurnDecision, ToolResult, TurnResultError, MessageDestination } from '../events'
import { catalog, isToolKey, type ToolKey } from '../catalog'
// ToolResult type is imported from ../events above
import { buildRegisteredTools, generateToolGrammar } from '../tools/tool-registry'

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
import { UNCLOSED_THINK_REMINDER, formatTaskOutsideSubtreeError } from '../prompts/error-states'
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
import { ReplayProjection } from '../projections/replay'
import { TaskGraphProjection, type TaskGraphState, type TaskStatus } from '../projections/task-graph'

import type { RoleDefinition, BoundObservable } from '@magnitudedev/roles'
import { bindObservable } from '@magnitudedev/roles'
import { ProjectionReaderTag, type ProjectionReader } from '../observables/projection-reader'
import { EphemeralSessionContextTag, PolicyContextProviderTag, type EphemeralSessionContext, type PolicyContext } from '../agents/types'
import { createPolicyContextProvider } from '../agents/policy-context'
import { ExecutionManager, IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from './types'
import type { TurnEvent, TurnEventSink, ExecuteOptions, ExecuteResult, ExecutionManagerService } from './types'
import { WorkingDirectoryTag } from './working-directory'
import type { StreamingLeaf, StreamHook, ToolDefinition } from '@magnitudedev/tools'


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
 * No sandboxes — xml-act runtime is created fresh per execute() call.
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
    execute: (xmlStream: Stream.Stream<string, ModelError>, options, sink) => Effect.gen(function* () {
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

      // Build registered tools for xml-act runtime
      const registeredTools = buildRegisteredTools(options.toolSet, executionLayer)

      // Create fresh runtime for this execution
      // Surface validation errors as TurnEngineCrash so they appear as turn errors
      const runtime = yield* Effect.try({
        try: () => createTurnEngine({
          tools: registeredTools,
          defaultProseDest,
          resultsDir: path.join(workspacePath, 'results'),
        }),
        catch: (e) => new TurnEngineCrash(`Runtime initialization failed: ${e instanceof Error ? e.message : String(e)}`, e),
      })

      // Get replay state from projection for crash recovery
      const replayProjection = yield* ReplayProjection.Tag
      const replayState: EngineState = yield* replayProjection.getFork(forkId)

      // Build tool tagName → defKey lookup from registered metadata.
      const tagToDefKey = new Map<string, string>()
      for (const [tagName, registered] of registeredTools.entries()) {
        const meta = registered.meta as Record<string, unknown> | undefined
        const defKey = typeof meta?.defKey === 'string' ? meta.defKey : undefined
        if (!defKey) continue
        tagToDefKey.set(tagName, defKey)
      }

      /** Resolve a xml-act event's tagName to the definition key. */
      const resolveKey = (tagName: string): ToolKey | undefined => {
        const key = tagToDefKey.get(tagName)
        return key && isToolKey(key) ? key : undefined
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

      // Track cached tool calls (replay) — skip their events
      const cachedToolCallIds = new Set<string>()

      // Position counter for tool events
      let positionCounter = 0

      // Streaming hook state per tool call
      const streamHookStates = new Map<string, unknown>()
      const streamHookConfigs = new Map<string, AnyStreamHook>()
      // Simple field accumulator for streaming partial input
      const streamingFields = new Map<string, Record<string, StreamingLeaf<unknown>>>()

      // Track tag names for ToolStarted events
      const toolCallTagNames = new Map<string, string>()

      const turnErrors: TurnResultError[] = []

      // Store execution result
      let executionResult: TurnResult = { success: true, turnDecision: 'idle' }

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


      // Capture raw model response text for identical-response circuit breaker.
      let rawResponseText = ''
      const trackedXmlStream = xmlStream.pipe(
        Stream.tap((chunk) => Effect.sync(() => {
          rawResponseText += chunk
        })),
      )

      // Run xml-act runtime
      const eventStream = runtime.streamWith(trackedXmlStream, { initialState: replayState })

      // Track toolCallId → internal catalog key for app events / projections.
      // xml-act events still carry the model-facing XML tagName; downstream
      // renderers should use that tagName when presenting tool errors back to the model.
      const toolCallKeys = new Map<string, ToolKey>()

      yield* Effect.scoped(
        eventStream.pipe(
          Stream.provideLayer(executionLayer),
          Stream.runForEach((event: TurnEngineEvent) => Effect.gen(function* () {
            switch (event._tag) {
              // --- Tool Input Started ---
              case 'ToolInputStarted': {
                hasAnyResponseContent = true
                const toolKey = resolveKey(event.tagName)
                if (!toolKey) {
                  logger.error(`[ExecutionManager] Failed to resolve tool key for tag ${event.tagName} (toolCallId: ${event.toolCallId}).`)
                  break
                }
                toolCallTagNames.set(event.toolCallId, event.tagName)
                toolCallKeys.set(event.toolCallId, toolKey)

                // Check for stream hook
                const toolEntry = catalog.entries[toolKey]
                const tool = toolEntry?.tool
                if (tool && hasStreamHook(tool)) {
                  const streamConfig = tool.stream
                  streamHookStates.set(event.toolCallId, streamConfig.initial)
                  streamHookConfigs.set(event.toolCallId, streamConfig)
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

                const streamConfig = streamHookConfigs.get(event.toolCallId)
                if (streamConfig) {
                  const currentState = streamHookStates.get(event.toolCallId)
                  const partialInput = streamingFields.get(event.toolCallId) ?? {}
                  const streamCtx = {
                    emit: (value: unknown) => sink.emit( {
                      _tag: 'ToolEvent' as const,
                      toolCallId: event.toolCallId,
                      toolKey,
                      event: { _tag: 'ToolEmission' as const, toolCallId: event.toolCallId, value },
                    }),
                  }
                  const streamEffect = streamConfig.onInput(partialInput, currentState, streamCtx) as Effect.Effect<unknown, unknown, any>
                  const newState = yield* (streamEffect.pipe(
                    Effect.provide(executionLayer),
                    Effect.catchAll(() => Effect.succeed(currentState)),
                  ) as Effect.Effect<unknown, never, never>)
                  streamHookStates.set(event.toolCallId, newState)
                }

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Input Parse Error ---
              case 'ToolInputParseError': {
                hasAnyResponseContent = true
                const toolKey = resolveKey(event.tagName)
                if (!toolKey) break
                toolCallKeys.set(event.toolCallId, toolKey)

                // Track for turn policy so the loop continues and LLM sees the error
                toolsCalledKeys.push(toolKey)
                lastToolKey = toolKey

                const errorResult: ToolResult = { status: 'error', message: event.error.detail }
                if (errorResult.status === 'error') {
                  hasToolErrors = true
                }

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Execution Started (check for cached/replay) ---
              case 'ToolExecutionStarted': {
                if (event.cached) {
                  cachedToolCallIds.add(event.toolCallId)
                }

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

                // Skip cached tool calls — these are replays
                if (cachedToolCallIds.has(event.toolCallId)) {
                  cachedToolCallIds.delete(event.toolCallId)
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
                const streamConfig = streamHookConfigs.get(event.toolCallId)
                if (streamConfig) {
                  const currentState = streamHookStates.get(event.toolCallId)
                  const partialInput = streamingFields.get(event.toolCallId) ?? {}
                  const streamCtx = {
                    emit: (value: unknown) => sink.emit( {
                      _tag: 'ToolEvent' as const,
                      toolCallId: event.toolCallId,
                      toolKey,
                      event: { _tag: 'ToolEmission' as const, toolCallId: event.toolCallId, value },
                    }),
                  }
                  const streamEffect = streamConfig.onInput(partialInput, currentState, streamCtx) as Effect.Effect<unknown, unknown, any>
                  const newState = yield* (streamEffect.pipe(
                    Effect.provide(executionLayer),
                    Effect.catchAll(() => Effect.succeed(currentState)),
                  ) as Effect.Effect<unknown, never, never>)
                  streamHookStates.set(event.toolCallId, newState)
                }

                yield* sink.emit( {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              case 'ToolInputFieldComplete':
                // Field complete — final value comes via ToolInputReady; emit for consumers
                {
                  const toolKey = toolCallKeys.get(event.toolCallId)
                  if (toolKey) {
                    yield* sink.emit( {
                      _tag: 'ToolEvent',
                      toolCallId: event.toolCallId,
                      toolKey,
                      event,
                    })
                  }
                }
                break

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
                      turnErrors.push({
                        code: 'nonexistent_agent_destination',
                        message: `Invalid message destination "${explicitTo}": only root fork can send to user`,
                      })
                      break
                    }
                    destination = { kind: 'user' }
                  } else if (explicitTo === 'parent') {
                    if (forkId === null) {
                      turnErrors.push({
                        code: 'nonexistent_agent_destination',
                        message: `Invalid message destination "${explicitTo}": root fork has no parent`,
                      })
                      break
                    }
                    destination = { kind: 'parent' }
                  } else {
                    const targetTask = taskState.tasks.get(explicitTo)
                    if (!targetTask) {
                      turnErrors.push({
                        code: 'nonexistent_agent_destination',
                        message: `Invalid message destination "${explicitTo}": task not found`,
                      })
                      break
                    }
                    if (!targetTask.worker) {
                      turnErrors.push({
                        code: 'nonexistent_agent_destination',
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
                    turnErrors.push({
                      code: 'task_operation_error',
                      message: messageResult.error,
                    })
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
                yield* sink.emit( { _tag: 'MessageChunk', id: event.id, text: event.text })
                break
              }

              case 'MessageEnd': {
                yield* sink.emit( { _tag: 'MessageEnd', id: event.id })
                break
              }

              case 'ProseChunk': {
                hasAnyResponseContent = true
                // ProseChunk is raw text content — lenses handle think blocks separately
                break
              }

              case 'ProseEnd': {
                hasAnyResponseContent = true
                // ProseEnd marks end of prose section — lenses handle think blocks separately
                break
              }

              case 'LensStart': {
                hasAnyResponseContent = true
                yield* sink.emit( { _tag: 'LensStarted', name: event.name })
                break
              }

              case 'LensChunk': {
                hasAnyResponseContent = true
                yield* sink.emit( { _tag: 'LensDelta', text: event.text })
                break
              }

              case 'LensEnd': {
                hasAnyResponseContent = true
                yield* sink.emit( { _tag: 'LensEnded', name: event.name })
                break
              }


              case 'ToolObservation': {
                hasAnyResponseContent = true
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



              case 'StructuralParseError': {
                if (event.error._tag === 'UnclosedThink') {
                  turnErrors.push({ code: 'unclosed_think', message: UNCLOSED_THINK_REMINDER })
                }
                break
              }

              // --- Turn End ---
              case 'TurnEnd': {
                const endResult = event.result
                if (endResult._tag === 'Success') {
                  // 'Success' here is a bit misleading — it only means the xml-act runtime finished
                  // parsing the stream without crashing. Tool errors, invalid refs, etc. are
                  // all considered "successful" executions at the runtime level.
                  //
                  // Errors within the turn always force continuation so the agent sees
                  // the error feedback and can retry. The turn policy only decides for clean turns.
                  if (hasToolErrors || turnErrors.length > 0) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      ...(turnErrors.length > 0 ? { errors: turnErrors } : {}),
                    }
                  } else if (endResult.turnControl === null && !hasAnyResponseContent) {
                    // Empty LLM response (no messages/tools/think/lens output).
                    // Always retrigger so memory-injected corrective feedback is visible next turn.
                    executionResult = { success: true, turnDecision: 'continue' }
                  } else {
                    // Map yield target to turn decision
                    // yield-invoke → continue (wait for tool results)
                    // yield-user, yield-worker, yield-parent → idle
                    const target = endResult.turnControl?.target ?? null
                    if (target === 'invoke') {
                      executionResult = { success: true, turnDecision: 'continue' }
                    } else {
                      // user, worker, parent, or null → idle
                      executionResult = { success: true, turnDecision: 'idle' }
                    }

                    // Apply turn policy when no explicit yield (null case)
                    if (target === null) {
                      const policyCtx = yield* policyCtxProvider.get
                      const turnResult = agentDef.getTurn({
                        toolsCalled: toolsCalledKeys,
                        lastTool: lastToolKey,
                        messagesSent,
                        state: policyCtx,
                      })
                      if (turnResult.action === 'continue') {
                        executionResult = { success: true, turnDecision: 'continue' }
                      } else {
                        executionResult = { success: true, turnDecision: 'idle' }
                      }
                    }
                  }
                } else if (endResult._tag === 'Interrupted') {
                  executionResult = { success: false, error: 'Interrupted', cancelled: true }
                } else if (endResult._tag === 'Failure') {
                  executionResult = { success: false, error: endResult.error, cancelled: false }
                } else if (endResult._tag === 'GateRejected') {
                  const rejection = endResult.rejection
                  if (rejection && typeof rejection === 'object' && '_tag' in rejection) {
                    const reason = 'reason' in rejection && typeof rejection.reason === 'string'
                      ? rejection.reason
                      : 'Gate rejected'
                    const cancelled = rejection._tag === 'UserRejection'
                    executionResult = { success: false, error: reason, cancelled }
                  } else {
                    executionResult = { success: false, error: String(rejection) || 'Gate rejected', cancelled: true }
                  }
                }

                // Circuit breaker: stop tight loops of identical consecutive continue responses.
                if (executionResult.success && executionResult.turnDecision === 'continue') {
                  const previous = identicalContinueTracker.get(forkId)
                  const nextCount = previous && previous.lastResponseText === rawResponseText
                    ? previous.consecutiveCount + 1
                    : 1

                  identicalContinueTracker.set(forkId, {
                    lastResponseText: rawResponseText,
                    consecutiveCount: nextCount,
                  })

                  if (nextCount >= IDENTICAL_RESPONSE_BREAKER_THRESHOLD) {
                    executionResult = {
                      success: false,
                      error: `Circuit breaker tripped after ${nextCount} identical consecutive responses.`,
                      cancelled: false,
                    }
                    identicalContinueTracker.delete(forkId)
                  }
                } else {
                  identicalContinueTracker.delete(forkId)
                }

                // yield-worker retrigger guard: if lead yields to workers but none are active, retrigger
                if (executionResult.success && executionResult.turnDecision === 'idle' && endResult._tag === 'Success' && endResult.turnControl?.target === 'worker') {
                  const pCtx = yield* policyCtxProvider.get
                  if (pCtx.activeAgentCount === 0) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      yieldWorkerRetriggered: true,
                    }
                  }
                }

                // Oneshot liveness guard: prevent stalling when nothing is active
                if (executionResult.success && executionResult.turnDecision === 'idle' && oneshotEnabled) {
                  const pCtx = yield* policyCtxProvider.get
                  if (pCtx.activeAgentCount === 0) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      oneshotLivenessTriggered: true,
                    }
                  }
                }
                break
              }
            }
          }))
        )
      )

      return {
        result: executionResult,
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