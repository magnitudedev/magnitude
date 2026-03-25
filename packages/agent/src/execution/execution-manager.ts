/**
 * ExecutionManager
 *
 * Owns per-fork lifecycle and xml-act runtime execution.
 * Maps XmlRuntimeEvents to agent TurnEvents.
 *
 * No sandboxes, no journals, no WASM — just xml-act streaming runtime.
 */

import { Effect, Stream, Queue, Context, Layer, Ref } from 'effect'
import type { ModelError } from '@magnitudedev/providers'
import {
  createXmlRuntime,
  ToolInterceptorTag,
  XmlRuntimeCrash,
  type XmlRuntimeEvent,
  type ReactorState,
  type ToolInterceptor,
  type OutputNode,
} from '@magnitudedev/xml-act'
import { Fork, Projection, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent, TurnResult, TurnDecision, TurnToolCall, ToolResult, ObservedResult } from '../events'
import { isToolKey, type ToolKey } from '../tools/tool-definitions'
import type { XmlToolResult } from '@magnitudedev/xml-act'
import { buildRegisteredTools } from '../tools'
import { defaultXmlTagName } from '../tools'
import { getAgentDefinition, isValidVariant, type AgentVariant } from '../agents'
import { buildPermissionInterceptor, type AgentResolver } from './permission-gate'
import { createApprovalState, ApprovalStateTag, type ApprovalStateService } from './approval-state'

import { BrowserService } from '../services/browser-service'
import { BrowserHarnessTag } from '../tools/browser-tools'

import { AgentStateReaderTag, type AgentStateReader } from '../tools/fork'
import { AgentRegistryStateReaderTag, type AgentRegistryStateReader } from '../tools/agent-registry-reader'
import { buildCloneContext, buildSpawnContext, UNCLOSED_THINK_REMINDER, UNCLOSED_ACTIONS_REMINDER, ONESHOT_LIVENESS_REMINDER, formatNonexistentAgentError } from '../prompts'
import type { JsonSchema } from '@magnitudedev/llm-core'
import { SkillStateReaderTag, type SkillStateReader } from '../tools/skill'
import { ConversationStateReaderTag, type ConversationStateReader } from '../tools/memory-reader'
import { WorkflowStateReaderTag, type WorkflowStateReader } from '../tools/workflow-reader'
import { ConversationProjection, type ConversationState } from '../projections/conversation'
import { createId } from '../util/id'
import { logger } from '@magnitudedev/logger'

import { AgentRoutingProjection, type AgentRoutingState, isActiveRoute, getRoutingEntryByForkId } from '../projections/agent-routing'
import { AgentStatusProjection, type AgentStatusState, getActiveAgent, getAgentByForkId } from '../projections/agent-status'
import { WorkingStateProjection, type ForkWorkingState } from '../projections/working-state'
import { SessionContextProjection, type SessionContextState } from '../projections/session-context'
import { ReplayProjection } from '../projections/replay'
import { WorkflowProjection, type WorkflowCriteriaState } from '../projections/workflow'
import { BackgroundProcessesProjection, getProcessesForFork, type BackgroundProcessesState } from '../projections/background-processes'

import type { RoleDefinition, ToolSet, BoundObservable } from '@magnitudedev/roles'
import { bindObservable } from '@magnitudedev/roles'
import { ProjectionReaderTag, type ProjectionReader } from '../observables/projection-reader'
import { EphemeralSessionContextTag, PolicyContextProviderTag, type EphemeralSessionContext, type PolicyContext } from '../agents/types'
import { createPolicyContextProvider } from '../agents/policy-context'
import type { TurnEvent } from './types'
import { ToolReminderTag } from './tool-reminder'
import { WorkingDirectoryTag } from './working-directory'
import { ToolExecutionContextTag } from './tool-execution-context'
import type { Tool, AnyTool, StreamingLeaf, StreamingPartial } from '@magnitudedev/tools'


import { ChatPersistence } from '../persistence/chat-persistence-service'
import { BackgroundProcessRegistryTag, make as makeBackgroundProcessRegistry } from '../processes/background-process-registry'

const { ForkContext } = Fork

type AgentDef = RoleDefinition<ToolSet, string, PolicyContext>

import { mapXmlToolResult } from '../util/tool-result'

// =============================================================================
// Types
// =============================================================================

export interface ExecuteOptions {
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly defaultProseDest: 'user' | 'parent'
  readonly allowSingleUserReplyThisTurn: boolean
}

export interface ExecuteResult {
  readonly result: TurnResult
  readonly toolCalls: readonly TurnToolCall[]
  readonly observedResults: readonly ObservedResult[]
}

// =============================================================================
// Service Interface
// =============================================================================

export interface ExecutionManagerService {
  /**
   * Execute an XML stream from the LLM.
   * XmlRuntimeEvents are mapped to TurnEvents and offered to the sink queue.
   * Returns the accumulated execution result.
   */
  readonly execute: (
    xmlStream: Stream.Stream<string, ModelError>,
    options: ExecuteOptions,
    sink: Queue.Queue<TurnEvent>,
  ) => Effect.Effect<
    ExecuteResult,
    XmlRuntimeCrash,
    Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ReactorState> | Projection.ForkedProjectionInstance<ForkWorkingState>
  >

  /**
   * Initialize a fork with the given agent variant.
   * Builds and caches the fork's Effect layers and bound observables.
   * No runtime object is created — xml-act runtime is built fresh per execute() call.
   */
  readonly initFork: (
    forkId: string | null,
    variant: AgentVariant
  ) => Effect.Effect<
    void,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ForkWorkingState> | Projection.ForkedProjectionInstance<WorkflowCriteriaState> | Projection.ProjectionInstance<ConversationState> | Projection.ProjectionInstance<BackgroundProcessesState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

  /**
   * Dispose a fork's cached state.
   */
  readonly disposeFork: (forkId: string) => Effect.Effect<void>

  /** Kill tracked background processes for a fork, or all processes for the root fork. */
  readonly interruptProcesses: (forkId: string | null) => Effect.Effect<void>

  /** Get bound observables for a fork */
  readonly getObservables: (forkId: string | null) => BoundObservable[]

  /** Flush buffered background-process output for the fork before observables run. */
  readonly flushProcesses: (forkId: string | null) => Effect.Effect<void>

  /**
   * Spawn a non-blocking background fork. Returns the forkId.
   */
  readonly fork: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    message?: string
    outputSchema?: JsonSchema | undefined
    mode: 'clone' | 'spawn'
    role: AgentVariant
    taskId: string
  }) => Effect.Effect<
    string,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<AgentRoutingState> | Projection.ProjectionInstance<AgentStatusState> | Projection.ForkedProjectionInstance<ForkWorkingState> | Projection.ForkedProjectionInstance<WorkflowCriteriaState> | Projection.ProjectionInstance<ConversationState> | Projection.ProjectionInstance<BackgroundProcessesState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

  /**
   * Release the browser for a fork (called when fork goes idle).
   * No-op if the fork has no browser.
   */
  readonly releaseBrowserFork: (forkId: string) => Effect.Effect<void>

  /**
   * The approval state service instance.
   * Exposed so workers (ApprovalWorker) and projections can register handlers and resolve approvals.
   */
  readonly approvalState: ApprovalStateService


}

export class ExecutionManager extends Context.Tag('ExecutionManager')<
  ExecutionManager,
  ExecutionManagerService
>() {}

// =============================================================================
// Implementation
// =============================================================================

/**
 * Build the unified Effect layer for a fork — covers tool execution, interceptor, and emit.
 * Tools use reader services, interceptor uses PolicyContextProvider + ApprovalState.
 */
function makeForkLayers(
  forkId: string | null,

  sessionContextProjection: Projection.ProjectionInstance<SessionContextState>,
  agentProjection: Projection.ProjectionInstance<AgentRoutingState>,
  agentStatusProjection: Projection.ProjectionInstance<AgentStatusState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkWorkingState>,
  workflowProjection: Projection.ForkedProjectionInstance<WorkflowCriteriaState>,

  conversationProjection: Projection.ProjectionInstance<ConversationState>,
  approvalState: ApprovalStateService,
  persistenceLayer: Layer.Layer<ChatPersistence, never, never>,
  permissionInterceptor: ReturnType<typeof buildPermissionInterceptor>,
  toolReminderRef: Ref.Ref<string[]>,
  cwd: string,
  workspacePath: string,
  ephemeralSessionContext: EphemeralSessionContext,
  backgroundProcessRegistry: ReturnType<typeof makeBackgroundProcessRegistry>,
) {
  const agentRegistryStateReaderLayer = Layer.succeed(AgentRegistryStateReaderTag, {
    getState: () => agentStatusProjection.get
  } satisfies AgentRegistryStateReader)

  const skillStateReaderLayer = Layer.succeed(SkillStateReaderTag, {
    getUserSkills: () => Effect.map(
      sessionContextProjection.get,
      (state) => state.context?.skills ?? []
    )
  } satisfies SkillStateReader)

  const conversationStateReaderLayer = Layer.succeed(ConversationStateReaderTag, {
    getState: () => conversationProjection.get
  } satisfies ConversationStateReader)

  const workflowStateReaderLayer = Layer.succeed(WorkflowStateReaderTag, {
    getState: (forkId: string | null) => workflowProjection.getFork(forkId),
  } satisfies WorkflowStateReader)

  const agentStateReaderLayer = Layer.succeed(AgentStateReaderTag, {
    getAgentState: () => agentStatusProjection.get,
    getAgent: (agentId: string) => Effect.map(agentStatusProjection.get, (state) => state.agents.get(agentId)),
  } satisfies AgentStateReader)

  const policyCtxProvider = createPolicyContextProvider(
    forkId,
    cwd,
    workspacePath,
    ephemeralSessionContext,
    agentStatusProjection,
    workingStateProjection,
  )

  const toolReminderLayer = Layer.succeed(ToolReminderTag, {
    add: (text: string) => Ref.update(toolReminderRef, (arr) => [...arr, text])
  })

  const providedInterceptor: ToolInterceptor = {
    beforeExecute: (ctx) =>
      permissionInterceptor(ctx).pipe(
        Effect.provideService(ForkContext, { forkId }),
        Effect.provideService(PolicyContextProviderTag, policyCtxProvider),
        Effect.provideService(ApprovalStateTag, approvalState),
      ),
  }

  return Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId }),

    agentRegistryStateReaderLayer,
    conversationStateReaderLayer,
    workflowStateReaderLayer,
    skillStateReaderLayer,
    agentStateReaderLayer,


    Layer.succeed(ApprovalStateTag, approvalState),
    Layer.succeed(WorkingDirectoryTag, { cwd, workspacePath }),
    Layer.succeed(EphemeralSessionContextTag, ephemeralSessionContext),
    Layer.succeed(PolicyContextProviderTag, policyCtxProvider),
    Layer.succeed(ToolInterceptorTag, providedInterceptor),
    Layer.succeed(BackgroundProcessRegistryTag, backgroundProcessRegistry),
    toolReminderLayer,
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

  // Per-fork tool reminder refs (shared between layers and execute() event handler)
  const toolReminderRefs = new Map<string | null, Ref.Ref<string[]>>()


  // Bound observables map
  const boundObservables = new Map<string | null, BoundObservable[]>()

  // Session-level oneshot mode flag (set once during root fork init, never changes)
  let oneshotEnabled = false

  // Approval state for gated tool calls
  const approvalState = createApprovalState()
  let publishBackgroundProcessEvent: ((event: AppEvent) => void) | null = null
  const backgroundProcessRegistry = makeBackgroundProcessRegistry((event) => {
    publishBackgroundProcessEvent?.(event)
  })


  // Maps forkId → variant, populated when forks are created.
  const forkAgentVariants = new Map<string, AgentVariant>()

  // Pre-built teardown effects (captured at initFork time with services already provided)
  const forkTeardowns = new Map<string, Effect.Effect<void>>()

  // Pre-built idle release effects (repeatable; run each time fork goes idle)
  const forkIdleReleases = new Map<string, Effect.Effect<void>>()

  function isToolAny(value: AnyTool): value is Tool.Any {
    return 'bindings' in value
  }

  function hasNameAndOptionalGroup(value: AnyTool): value is Extract<AnyTool, { name: string; group?: string }> {
    return typeof value.name === 'string'
  }

  /**
   * Resolve the active agent definition for a fork.
   * Child forks use their fixed role. Root fork uses the orchestrator definition.
   */
  const resolveAgent: AgentResolver = (forkId) => {
    if (forkId !== null) {
      const variant = forkAgentVariants.get(forkId) ?? 'builder'
      return getAgentDefinition(variant)
    }
    return getAgentDefinition('lead')
  }

  // Build the permission interceptor (shared across all forks, resolves agent dynamically)
  const permissionInterceptor = buildPermissionInterceptor(resolveAgent)

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
      const { forkId, turnId, defaultProseDest, allowSingleUserReplyThisTurn } = options

      // Resolve agent definition for this fork
      const agentRoutingProjectionInst = yield* AgentRoutingProjection.Tag
      const agentStatusProjectionInst = yield* AgentStatusProjection.Tag
      const workingStateProjectionInst = yield* WorkingStateProjection.Tag
      const agentState = yield* agentStatusProjectionInst.get
      let variant: AgentVariant
      if (forkId) {
        const agentInstance = getAgentByForkId(agentState, forkId)
        const role = agentInstance?.role
        variant = role && isValidVariant(role) ? role : 'builder'
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

      const executionLayer = Layer.merge(
        layers,
        Layer.succeed(ToolExecutionContextTag, { turnId }),
      )

      // Build registered tools for xml-act runtime
      const registeredTools = buildRegisteredTools(agentDef, executionLayer)

      // Create fresh xml-act runtime for this execution
      // Surface binding validation errors as XmlRuntimeCrash so they appear as turn errors
      const runtime = yield* Effect.try({
        try: () => createXmlRuntime({
          tools: registeredTools,
          defaultProseDest,
        }),
        catch: (e) => new XmlRuntimeCrash(`XML binding validation failed: ${e instanceof Error ? e.message : String(e)}`, e),
      })

      // Get replay state from projection for crash recovery
      const replayProjection = yield* ReplayProjection.Tag
      const replayState: ReactorState = yield* replayProjection.getFork(forkId)

      // ToolReminder ref shared with tool services
      const toolReminderRef = toolReminderRefs.get(forkId)!


      // Build tool tagName → defKey lookup and tagName → tool lookup
      // Use registeredTools metadata so both legacy and defineTool tools are covered.
      const tagToDefKey = new Map<string, string>()
      const tagToTool = new Map<string, AnyTool>()
      for (const [tagName, registered] of registeredTools.entries()) {
        const meta = registered.meta as Record<string, unknown> | undefined
        const defKey = typeof meta?.defKey === 'string' ? meta.defKey : undefined
        if (!defKey) continue
        tagToDefKey.set(tagName, defKey)
        const tool = agentDef.tools[defKey]
        if (tool) {
          tagToTool.set(tagName, tool)
        }
      }

      /** Resolve a xml-act event's tagName to the definition key. */
      const resolveKey = (tagName: string): ToolKey | undefined => {
        const key = tagToDefKey.get(tagName) ?? tagName
        return isToolKey(key) ? key : undefined
      }

      // Also build callable-based lookup for resolveKey from group+toolName
      const callableToKey = new Map<string, string>()
      for (const [key, tool] of Object.entries(agentDef.tools)) {
        if (!tool || !hasNameAndOptionalGroup(tool)) continue
        const group = tool.group
        const callable = (group && group !== 'default') ? `${group}.${tool.name}` : tool.name
        callableToKey.set(callable, key)
      }
      const resolveKeyFromCallable = (group: string, toolName: string): ToolKey | undefined => {
        const callable = group === 'default' ? toolName : `${group}.${toolName}`
        const key = callableToKey.get(callable) ?? callable
        return isToolKey(key) ? key : undefined
      }

      // Track tools called (by definition key) for turn policy
      const toolsCalledKeys: ToolKey[] = []
      let lastToolKey: ToolKey | null = null
      const toolCalls: TurnToolCall[] = []
      const messagesSent: Array<{ id: string, dest: string }> = []
      let hasAnyMessage = false
      let directUserRepliesSent = 0

      // Track tool input (ToolInputReady provides the parsed input)
      const toolInputs = new Map<string, unknown>()

      // Track cached tool calls (replay) — skip their events
      const cachedToolCallIds = new Set<string>()

      // Position counter for tool events
      let positionCounter = 0

      // Streaming hook state per tool call
      const streamHookStates = new Map<string, unknown>()
      const streamHookConfigs = new Map<string, {
        onInput: (
          input: StreamingPartial<Record<string, unknown>>,
          state: unknown,
          ctx: { emit: (value: unknown) => Effect.Effect<void> }
        ) => Effect.Effect<unknown, unknown, unknown>
        initial: unknown
      }>()
      // Simple field accumulator for streaming partial input
      const streamingFields = new Map<string, Record<string, StreamingLeaf<unknown>>>()

      // Track tag names for ToolStarted events
      const toolCallTagNames = new Map<string, string>()

      // Accumulate observed tool output results
      const observedResults: ObservedResult[] = []

      let hasStructuralParseError = false
      let structuralParseErrorReminder: string | null = null
      const collectedToolReminders: string[] = []

      // Track messages to nonexistent agent IDs
      let hasNonexistentAgentDest = false
      let nonexistentAgentError: string | null = null

      // Store execution result
      let executionResult: TurnResult = { success: true, turnDecision: 'yield' }

      // Create the PolicyContextProvider for turn policy evaluation
      const cwd = forkCwds.get(forkId) ?? process.cwd()
      const workspacePath = forkWorkspacePaths.get(forkId)!

      const policyCtxProvider = createPolicyContextProvider(
        forkId,
        cwd,
        workspacePath,
        ephemeralSessionContext,
        yield* AgentStatusProjection.Tag,
        yield* WorkingStateProjection.Tag,
      )


      // Run xml-act runtime
      const eventStream = runtime.streamWith(xmlStream, { initialState: replayState })

      // Track toolCallId → toolKey mapping for event forwarding
      const toolCallKeys = new Map<string, ToolKey>()

      yield* Effect.scoped(
        eventStream.pipe(
          Stream.provideLayer(executionLayer),
          Stream.runForEach((event: XmlRuntimeEvent) => Effect.gen(function* () {
            switch (event._tag) {
              // --- Tool Input Started ---
              case 'ToolInputStarted': {
                const toolKey = resolveKeyFromCallable(event.group, event.toolName)
                if (!toolKey) break
                toolCallTagNames.set(event.toolCallId, event.tagName)
                toolCallKeys.set(event.toolCallId, toolKey)

                // Check for stream hook
                const tool = tagToTool.get(event.tagName)
                if (tool && 'stream' in tool && (tool as any).stream) {
                  const streamConfig = (tool as any).stream as {
                    onInput: (
                      input: StreamingPartial<Record<string, unknown>>,
                      state: unknown,
                      ctx: { emit: (value: unknown) => Effect.Effect<void> }
                    ) => Effect.Effect<unknown, unknown, unknown>
                    initial: unknown
                  }
                  streamHookStates.set(event.toolCallId, streamConfig.initial)
                  streamHookConfigs.set(event.toolCallId, streamConfig)
                  streamingFields.set(event.toolCallId, {})
                }

                yield* Queue.offer(sink, {
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

                const toolKey = toolCallKeys.get(event.toolCallId) ?? event.toolCallId

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
                    emit: (value: unknown) => Queue.offer(sink, {
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

                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Input Parse Error ---
              case 'ToolInputParseError': {
                const toolKey = resolveKeyFromCallable(event.group, event.toolName)
                if (!toolKey) break
                toolCallKeys.set(event.toolCallId, toolKey)

                // Track for turn policy so the loop continues and LLM sees the error
                toolsCalledKeys.push(toolKey)
                lastToolKey = toolKey

                const errorResult: ToolResult = { status: 'error', message: event.error.detail }
                toolCalls.push({ toolKey, group: event.group, toolName: event.toolName, result: errorResult })

                yield* Queue.offer(sink, {
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
                // Reset tool reminders for the new tool execution
                yield* Ref.set(toolReminderRef, [])

                const toolKey = toolCallKeys.get(event.toolCallId) ?? resolveKeyFromCallable(event.group, event.toolName)
                if (!toolKey) break
                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              case 'ToolEmission': {
                const toolKey = toolCallKeys.get(event.toolCallId) ?? event.toolCallId
                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool Execution Ended ---
              case 'ToolExecutionEnded': {
                const toolKey = resolveKeyFromCallable(event.group, event.toolName)
                if (!toolKey) break

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

                // Read and clear tool reminders
                const reminders = yield* Ref.getAndSet(toolReminderRef, [])

                // Map XmlToolResult → ToolResult for TurnToolCall accumulation
                const toolResult: ToolResult = mapXmlToolResult(event.result)

                toolCalls.push({
                  toolKey,
                  group: event.group,
                  toolName: event.toolName,
                  result: toolResult,
                })
                collectedToolReminders.push(...reminders)

                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Tool streaming events (field values, body chunks, children) ---
              case 'ToolInputBodyChunk':
              case 'ToolInputFieldValue':
              case 'ToolInputChildStarted':
              case 'ToolInputChildComplete': {
                const toolKey = toolCallKeys.get(event.toolCallId) ?? event.toolCallId

                // Accumulate fields for stream hook
                if (event._tag === 'ToolInputFieldValue') {
                  const fields = streamingFields.get(event.toolCallId)
                  if (fields) {
                    fields[event.field] = { value: event.value, isFinal: true }
                  }
                }

                if (event._tag === 'ToolInputChildStarted') {
                  const fields = streamingFields.get(event.toolCallId)
                  if (fields) {
                    fields[event.field] = { value: '', isFinal: false }
                  }
                }

                if (event._tag === 'ToolInputChildComplete') {
                  const fields = streamingFields.get(event.toolCallId)
                  if (fields && fields[event.field]) {
                    fields[event.field] = { value: fields[event.field]!.value, isFinal: true }
                  }
                }

                if (event._tag === 'ToolInputBodyChunk') {
                  const fields = streamingFields.get(event.toolCallId)
                  if (fields) {
                    const existing = fields._body
                    const prior = typeof existing?.value === 'string' ? existing.value : ''
                    fields._body = { value: `${prior}${event.text}`, isFinal: false }
                  }
                }

                // Invoke stream hook if present
                const streamConfig = streamHookConfigs.get(event.toolCallId)
                if (streamConfig) {
                  const currentState = streamHookStates.get(event.toolCallId)
                  const partialInput = streamingFields.get(event.toolCallId) ?? {}
                  const streamCtx = {
                    emit: (value: unknown) => Queue.offer(sink, {
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

                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                })
                break
              }

              // --- Messages / Think prose ---
              case 'MessageStart': {
                let resolvedDest = event.dest
                if (forkId !== null && event.dest === 'user') {
                  if (!allowSingleUserReplyThisTurn || directUserRepliesSent >= 1) {
                    resolvedDest = 'parent'
                  } else {
                    directUserRepliesSent += 1
                  }
                }

                messagesSent.push({ id: event.id, dest: resolvedDest })
                hasAnyMessage = true

                // Validate agent message destinations inline during execution
                if (resolvedDest !== 'user' && resolvedDest !== 'parent') {
                  const currentAgentState = yield* agentRoutingProjectionInst.get
                  const targetAgent = isActiveRoute(currentAgentState, resolvedDest)
                  if (!targetAgent) {
                    hasNonexistentAgentDest = true
                    const destStr = `"${resolvedDest}"`
                    nonexistentAgentError = `<error>\n${formatNonexistentAgentError(destStr)}\n</error>`
                  }
                }

                yield* Queue.offer(sink, { _tag: 'MessageStart', id: event.id, dest: resolvedDest })
                break
              }

              case 'MessageChunk': {
                yield* Queue.offer(sink, { _tag: 'MessageChunk', id: event.id, text: event.text })
                break
              }

              case 'MessageEnd': {
                yield* Queue.offer(sink, { _tag: 'MessageEnd', id: event.id })
                break
              }

              case 'ProseChunk': {
                if (event.patternId === 'think') {
                  yield* Queue.offer(sink, { _tag: 'ThinkingDelta', text: event.text })
                }
                break
              }

              case 'ProseEnd': {
                if (event.patternId === 'think') {
                  yield* Queue.offer(sink, { _tag: 'ThinkingEnd', about: event.about })
                }
                break
              }

              case 'LensStart': {
                yield* Queue.offer(sink, { _tag: 'LensStarted', name: event.name })
                break
              }

              case 'LensChunk': {
                yield* Queue.offer(sink, { _tag: 'LensDelta', text: event.text })
                break
              }

              case 'LensEnd': {
                yield* Queue.offer(sink, { _tag: 'LensEnded', name: event.name })
                break
              }


              case 'ToolObservation': {
                observedResults.push({
                  toolCallId: event.toolCallId,
                  tagName: event.tagName,
                  query: event.query,
                  content: event.content,
                })
                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey: toolCallKeys.get(event.toolCallId) ?? resolveKey(event.tagName) ?? 'shell',
                  event,
                })
                break
              }

              case 'StructuralParseError': {
                hasStructuralParseError = true
                const msg = event.error._tag === 'UnclosedThink'
                  ? UNCLOSED_THINK_REMINDER
                  : event.error._tag === 'FinishWithoutEvidence'
                  ? event.error.detail
                  : UNCLOSED_ACTIONS_REMINDER
                structuralParseErrorReminder = `<error>\n${msg}\n</error>`
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
                  const hasToolErrors = toolCalls.some(tc => tc.result.status === 'error')

                  if (hasToolErrors || hasStructuralParseError || hasNonexistentAgentDest) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      ...(structuralParseErrorReminder
                        ? { reminder: structuralParseErrorReminder }
                        : nonexistentAgentError
                          ? { reminder: nonexistentAgentError }
                          : {}),
                    }
                  } else if (endResult.turnControl === 'finish') {
                    executionResult = { success: true, turnDecision: 'finish', evidence: endResult.evidence }
                  } else if (endResult.turnControl) {
                    executionResult = { success: true, turnDecision: endResult.turnControl }
                  } else {
                    const policyCtx = yield* policyCtxProvider.get
                    const turnResult = agentDef.getTurn({
                      toolsCalled: toolsCalledKeys,
                      lastTool: lastToolKey,
                      messagesSent,
                      state: policyCtx,
                    })
                    if (turnResult.action === 'finish') {
                      executionResult = { success: true, turnDecision: 'finish', reminder: turnResult.reminder, evidence: '' }
                    } else {
                      executionResult = { success: true, turnDecision: turnResult.action, reminder: turnResult.reminder }
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

                if (executionResult.success && collectedToolReminders.length > 0) {
                  const existingReminder = executionResult.reminder
                  executionResult = {
                    ...executionResult,
                    reminder: existingReminder
                      ? `${existingReminder}\n${collectedToolReminders.join('\n')}`
                      : collectedToolReminders.join('\n'),
                  }
                }

                // Oneshot liveness guard: prevent stalling when nothing is active
                if (executionResult.success && executionResult.turnDecision === 'yield' && oneshotEnabled) {
                  const pCtx = yield* policyCtxProvider.get
                  if (pCtx.activeAgentCount === 0) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      reminder: ONESHOT_LIVENESS_REMINDER,
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
        toolCalls,
        observedResults,
      }
    }),

    initFork: (forkId, variant) => Effect.gen(function* () {
      const workerBus = yield* WorkerBusTag<AppEvent>()
      publishBackgroundProcessEvent = (event) => {
        void Effect.runPromise(workerBus.publish(event))
      }

      const sessionContextProjection = yield* SessionContextProjection.Tag
      const agentProjection = yield* AgentRoutingProjection.Tag
      const agentStatusProjection = yield* AgentStatusProjection.Tag
      const workingStateProjection = yield* WorkingStateProjection.Tag
      const workflowProjection = yield* WorkflowProjection.Tag

      const conversationProjection = yield* ConversationProjection.Tag
      const persistence = yield* ChatPersistence
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)

      // Create ToolReminder refs (shared with execute() event handler via maps)
      const toolReminderRef = yield* Ref.make<string[]>([])
      toolReminderRefs.set(forkId, toolReminderRef)
      const sessionState = yield* sessionContextProjection.get
      const cwd = sessionState.context?.cwd ?? process.cwd()
      const workspacePath = sessionState.context?.workspacePath ?? cwd
      if (forkId === null) {
        oneshotEnabled = !!sessionState.context?.oneshot
      }

      let layers = makeForkLayers(
        forkId,
        sessionContextProjection, agentProjection, agentStatusProjection,
        workingStateProjection, workflowProjection,
        conversationProjection,
        approvalState,
        persistenceLayer, permissionInterceptor, toolReminderRef, cwd, workspacePath, ephemeralSessionContext, backgroundProcessRegistry,
      )
      forkCwds.set(forkId, cwd)
      forkWorkspacePaths.set(forkId, workspacePath)

      // Inject role-specific setup layer when the role defines a setup function
      const roleDef = getAgentDefinition(variant)
      if (roleDef.setup && forkId) {
        const setupLayer = yield* roleDef.setup({ forkId, cwd, workspacePath })
        layers = Layer.merge(layers, setupLayer)
      }

      // Pre-build teardown effect with services captured now (so disposeFork needs no requirements)
      if (forkId && (roleDef.setup || roleDef.teardown)) {
        const browserService = yield* BrowserService

        if (roleDef.teardown) {
          const teardownEffect = roleDef.teardown({ forkId, cwd, workspacePath }).pipe(
            Effect.provideService(BrowserService, browserService)
          )
          forkTeardowns.set(forkId, teardownEffect)
        }

        // Store repeatable idle release (for browser forks)
        forkIdleReleases.set(forkId, browserService.release(forkId))
      }

      // Store variant for agent resolution
      if (forkId !== null) {
        forkAgentVariants.set(forkId, variant)
      }

      const backgroundProcessesProjection = yield* Effect.serviceOption(BackgroundProcessesProjection.Tag)

      const projectionReader: ProjectionReader = {
        getAgentRouting: () => agentProjection.get,
        getAgentStatus: () => agentStatusProjection.get,
        getBackgroundProcesses: () => backgroundProcessesProjection._tag === 'Some'
          ? Effect.map(backgroundProcessesProjection.value.get, (state) => getProcessesForFork(state, forkId))
          : Effect.succeed(new Map()),
      }
      const projectionReaderLayer = Layer.succeed(ProjectionReaderTag, projectionReader)
      layers = Layer.merge(layers, projectionReaderLayer)

      // Cache the layers
      forkLayers.set(forkId, layers)

      // Bind observables
      const agentDef = getAgentDefinition(variant)
      const agentObservables = agentDef.observables.map((obs) =>
        bindObservable(obs, () => Effect.succeed(layers))
      )
      boundObservables.set(forkId, agentObservables)
    }),

    disposeFork: (forkId) => Effect.gen(function* () {
      // Run role teardown if defined (e.g. browser cleanup)
      const teardown = forkTeardowns.get(forkId)
      if (teardown) {
        yield* Effect.ignore(teardown)
        forkTeardowns.delete(forkId)
      }

      yield* backgroundProcessRegistry.cleanupFork(forkId)
      forkLayers.delete(forkId)
      forkCwds.delete(forkId)
      forkWorkspacePaths.delete(forkId)
      toolReminderRefs.delete(forkId)
      boundObservables.delete(forkId)
      forkAgentVariants.delete(forkId)
      forkIdleReleases.delete(forkId)
    }),

    interruptProcesses: (forkId) => (
      forkId === null
        ? backgroundProcessRegistry.shutdownAll()
        : backgroundProcessRegistry.cleanupFork(forkId)
    ),

    fork: (params: {
      parentForkId: string | null
      name: string
      agentId: string
      prompt: string
      message?: string
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

      yield* workerBus.publish({
        type: 'agent_created',
        forkId,
        parentForkId: params.parentForkId,
        agentId: params.agentId,
        name: params.name,
        role: params.role,
        context,
        mode: params.mode,
        taskId: params.taskId ?? '',
        message: params.message ?? params.prompt,
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
    flushProcesses: (forkId) => backgroundProcessRegistry.flush(forkId),
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