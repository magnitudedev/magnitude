/**
 * ExecutionManager
 *
 * Owns per-fork lifecycle and xml-act runtime execution.
 * Maps XmlRuntimeEvents to agent TurnEvents.
 *
 * No sandboxes, no journals, no WASM — just xml-act streaming runtime.
 */

import { Effect, Stream, Queue, Context, Layer, Ref, Deferred } from 'effect'
import {
  createXmlRuntime,
  ToolInterceptorTag,
  XmlRuntimeCrash,
  type XmlRuntimeEvent,
  type ReactorState,
  type ToolInterceptor,
  type OutputNode,
  outputToText,
} from '@magnitudedev/xml-act'
import { Fork, Projection, WorkerBusTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent, TurnResult, TurnDecision, TurnToolCall, ToolResult, ToolDisplay, InspectResult } from '../events'
import type { XmlToolResult } from '@magnitudedev/xml-act'
import { buildRegisteredTools } from '../tools'
import { defaultXmlTagName } from '../tools'
import { getAgentDefinition, type AgentVariant } from '../agents'
import { buildPermissionInterceptor, type AgentResolver } from './permission-gate'
import { createApprovalState, ApprovalStateTag, type ApprovalStateService } from './approval-state'

import { BrowserService } from '../services/browser-service'
import { BrowserHarnessTag } from '../tools/browser-tools'

import { ForkStateReaderTag, type ForkStateReader } from '../tools/fork'
import { ArtifactStateReaderTag, type ArtifactStateReader } from '../tools/artifact-tools'
import { AgentRegistryStateReaderTag, type AgentRegistryStateReader } from '../tools/agent-registry-reader'
import { buildCloneContext, buildSpawnContext, UNCLOSED_THINK_REMINDER, UNCLOSED_ACTIONS_REMINDER, UNCLOSED_INSPECT_REMINDER, formatNonexistentAgentError } from '../prompts'
import type { JsonSchema } from '@magnitudedev/llm-core'
import { SkillStateReaderTag, type SkillStateReader } from '../tools/skill'
import { ConversationStateReaderTag, type ConversationStateReader } from '../tools/memory-reader'
import { ConversationProjection, type ConversationState } from '../projections/conversation'
import { createId } from '../util/id'
import { logger } from '@magnitudedev/logger'

import { ArtifactProjection, type ArtifactState } from '../projections/artifact'
import { AgentRegistryProjection, type AgentRegistryState } from '../projections/agent-registry'
import { WorkingStateProjection, type ForkWorkingState } from '../projections/working-state'
import { ForkProjection, type ForkState } from '../projections/fork'
import { SessionContextProjection, type SessionContextState } from '../projections/session-context'
import { ReplayProjection } from '../projections/replay'

import type { AgentDefinition, ToolSet, BoundObservable } from '@magnitudedev/agent-definition'
import { bindObservable } from '@magnitudedev/agent-definition'
import { ProjectionReaderTag, type ProjectionReader } from '../observables/projection-reader'
import { PolicyContextProviderTag, type PolicyContext } from '../agents/types'
import { createPolicyContextProvider } from '../agents/policy-context'
import type { TurnEvent } from './types'
import { ToolEmitTag } from './tool-emit'
import { WorkingDirectoryTag } from './working-directory'
import type { Tool } from '@magnitudedev/tools'


import { ChatPersistence } from '../persistence/chat-persistence-service'

const { ForkContext } = Fork

type AgentDef = AgentDefinition<ToolSet, PolicyContext>

/** Map XmlToolResult → ToolResult (display convenience type). */
function mapXmlToolResult(result: XmlToolResult, display?: ToolDisplay): ToolResult {
  switch (result._tag) {
    case 'Success':
      return { status: 'success', output: result.output, ...(display ? { display } : {}) }
    case 'Error':
      return { status: 'error', message: result.error }
    case 'Rejected': {
      const rej = result.rejection
      const isPerm = rej && typeof rej === 'object' && '_tag' in rej
      if (isPerm) {
        const r = rej as { _tag: string; reason: string }
        if (r._tag === 'UserRejection') {
          return { status: 'rejected', message: 'User rejected the action' }
        }
        return { status: 'rejected', message: 'System rejected', reason: r.reason }
      }
      return { status: 'rejected', message: String(rej) }
    }
    case 'Interrupted':
      return { status: 'interrupted' }
  }
}

// =============================================================================
// Types
// =============================================================================

export interface ExecuteOptions {
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
}

export interface ExecuteResult {
  readonly result: TurnResult
  readonly code: string
  readonly toolCalls: readonly TurnToolCall[]
  readonly inspectResults: readonly InspectResult[]
  readonly syntheticInspectCode?: string
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
    xmlStream: Stream.Stream<string, Error>,
    options: ExecuteOptions,
    sink: Queue.Queue<TurnEvent>,
  ) => Effect.Effect<
    ExecuteResult,
    XmlRuntimeCrash,
    Projection.ProjectionInstance<ForkState> | Projection.ForkedProjectionInstance<ReactorState> | Projection.ProjectionInstance<AgentRegistryState> | Projection.ForkedProjectionInstance<ForkWorkingState>
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
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<ForkState> | Projection.ProjectionInstance<ArtifactState> | Projection.ProjectionInstance<AgentRegistryState> | Projection.ForkedProjectionInstance<ForkWorkingState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService
  >

  /**
   * Dispose a fork's cached state.
   */
  readonly disposeFork: (forkId: string) => Effect.Effect<void>

  /** Get bound observables for a fork */
  readonly getObservables: (forkId: string | null) => BoundObservable[]

  /**
   * Spawn a non-blocking background fork. Returns the forkId.
   */
  readonly fork: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    outputSchema?: unknown
    mode: 'clone' | 'spawn'
    role: AgentVariant
    taskId: string
  }) => Effect.Effect<
    string,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<ForkState> | Projection.ProjectionInstance<ArtifactState> | Projection.ProjectionInstance<AgentRegistryState> | Projection.ForkedProjectionInstance<ForkWorkingState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

  /**
   * Spawn a blocking fork. Blocks until the fork calls submit(), returns the parsed result.
   */
  readonly forkSync: (params: {
    parentForkId: string | null
    name: string
    agentId: string
    prompt: string
    outputSchema?: unknown
    mode: 'clone' | 'spawn'
    role: AgentVariant
    taskId: string
    timeLimit?: number
  }) => Effect.Effect<
    unknown,
    never,
    Projection.ProjectionInstance<SessionContextState> | Projection.ProjectionInstance<ForkState> | Projection.ProjectionInstance<ArtifactState> | Projection.ProjectionInstance<AgentRegistryState> | Projection.ForkedProjectionInstance<ForkWorkingState> | Projection.ProjectionInstance<ConversationState> | ChatPersistence | BrowserService | WorkerBusService<AppEvent>
  >

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
  forkProjection: Projection.ProjectionInstance<ForkState>,
  artifactProjection: Projection.ProjectionInstance<ArtifactState>,
  agentRegistryProjection: Projection.ProjectionInstance<AgentRegistryState>,
  workingStateProjection: Projection.ForkedProjectionInstance<ForkWorkingState>,

  conversationProjection: Projection.ProjectionInstance<ConversationState>,
  blockingDeferreds: Map<string, Deferred.Deferred<unknown, never>>,


  approvalState: ApprovalStateService,
  persistenceLayer: Layer.Layer<ChatPersistence, never, never>,
  rawBeforeExecute: ReturnType<typeof buildPermissionInterceptor>,
  toolEmitRef: Ref.Ref<ToolDisplay | undefined>,
  cwd: string,
) {
  const artifactStateReaderLayer = Layer.succeed(ArtifactStateReaderTag, {
    getState: () => artifactProjection.get
  } satisfies ArtifactStateReader)

  const agentRegistryStateReaderLayer = Layer.succeed(AgentRegistryStateReaderTag, {
    getState: () => agentRegistryProjection.get
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

  const forkStateReaderLayer = Layer.succeed(ForkStateReaderTag, {
    getForkState: () => forkProjection.get,
    registerBlocking: (forkId, deferred) => { blockingDeferreds.set(forkId, deferred) },
    resolveBlocking: (forkId, result) => {
      const deferred = blockingDeferreds.get(forkId)
      if (!deferred) return Effect.void
      blockingDeferreds.delete(forkId)
      return Deferred.succeed(deferred, result)
    }
  } satisfies ForkStateReader)

  const policyCtxProvider = createPolicyContextProvider(forkId, cwd, agentRegistryProjection, workingStateProjection)

  const toolEmitLayer = Layer.succeed(ToolEmitTag, {
    emit: (value: ToolDisplay) => Ref.set(toolEmitRef, value)
  })

  // Wrap the raw interceptor (which has service requirements) into a ToolInterceptor (R=never)
  // by providing the services inline so the Effect has no remaining requirements.
  const interceptor: ToolInterceptor = {
    beforeExecute: (ctx) =>
      rawBeforeExecute(ctx).pipe(
        Effect.provideService(ForkContext, { forkId }),
        Effect.provideService(PolicyContextProviderTag, policyCtxProvider),
        Effect.provideService(ApprovalStateTag, approvalState),
      ),
  }

  return Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId }),

    artifactStateReaderLayer,
    agentRegistryStateReaderLayer,
    conversationStateReaderLayer,
    skillStateReaderLayer,
    forkStateReaderLayer,


    Layer.succeed(ApprovalStateTag, approvalState),
    Layer.succeed(WorkingDirectoryTag, { cwd }),
    Layer.succeed(PolicyContextProviderTag, policyCtxProvider),
    Layer.succeed(ToolInterceptorTag, interceptor),
    toolEmitLayer,
    persistenceLayer,
  )
}

/**
 * Create the execution manager.
 * No sandboxes — xml-act runtime is created fresh per execute() call.
 */
const makeExecutionManager = Effect.gen(function* () {
  // Per-fork cached layers (built during initFork, reused across turns)
  const forkLayers = new Map<string | null, Layer.Layer<never>>()
  const forkCwds = new Map<string | null, string>()

  // Per-fork tool emit refs (shared between layers and execute() event handler)
  const toolEmitRefs = new Map<string | null, Ref.Ref<ToolDisplay | undefined>>()

  // Bound observables map
  const boundObservables = new Map<string | null, BoundObservable[]>()

  // Shared map for blocking fork deferreds (forkSync registers, submit resolves)
  const blockingDeferreds = new Map<string, Deferred.Deferred<unknown, never>>()

  // Approval state for gated tool calls
  const approvalState = createApprovalState()


  // Maps forkId → variant, populated when forks are created.
  const forkAgentVariants = new Map<string, AgentVariant>()

  /**
   * Resolve the active agent definition for a fork.
   * Child forks use their fixed role. Root fork is always orchestrator.
   */
  const resolveAgent: AgentResolver = (forkId) => {
    if (forkId !== null) {
      const variant = forkAgentVariants.get(forkId) ?? 'builder'
      return getAgentDefinition(variant)
    }
    return getAgentDefinition('orchestrator')
  }

  // Build the permission interceptor (shared across all forks, resolves agent dynamically)
  const permissionInterceptor = buildPermissionInterceptor(resolveAgent)

  function buildForkContext(params: { mode: string; prompt: string; outputSchema?: unknown }) {
    return Effect.gen(function* () {
      if (params.mode === 'clone') {
        return buildCloneContext(params.prompt, params.outputSchema as JsonSchema | undefined)
      }
      const proj = yield* SessionContextProjection.Tag
      const ctx = yield* Effect.map(proj.get, s => s.context)
      return buildSpawnContext(params.prompt, ctx, params.outputSchema as JsonSchema | undefined)
    })
  }

  const service: ExecutionManagerService = {
    execute: (xmlStream, options, sink) => Effect.gen(function* () {
      const { forkId, turnId } = options

      // Resolve agent definition for this fork
      const forkProjectionInst = yield* ForkProjection.Tag
      const forkState = yield* forkProjectionInst.get
      let variant: AgentVariant
      if (forkId) {
        const forkInstance = forkState.forks.get(forkId)
        variant = (forkInstance?.role ?? 'builder') as AgentVariant
      } else {
        variant = 'orchestrator'
      }
      const agentDef = getAgentDefinition(variant)

      // Get cached fork layers (must be initialized via initFork)
      const layers = forkLayers.get(forkId)
      if (!layers) {
        return yield* Effect.die(
          new Error(`Fork not initialized: ${forkId}. initFork() must be called before execute().`)
        )
      }

      // Build registered tools for xml-act runtime
      const registeredTools = buildRegisteredTools(agentDef, layers)

      // Create fresh xml-act runtime for this execution
      // Surface binding validation errors as XmlRuntimeCrash so they appear as turn errors
      const runtime = yield* Effect.try({
        try: () => createXmlRuntime({
          tools: registeredTools,
          defaultProseDest: forkId !== null ? 'parent' : 'user',
        }),
        catch: (e) => new XmlRuntimeCrash(`XML binding validation failed: ${e instanceof Error ? e.message : String(e)}`, e),
      })

      // Get replay state from projection for crash recovery
      const replayProjection = yield* ReplayProjection.Tag
      const replayState: ReactorState = yield* replayProjection.getFork(forkId)

      // ToolEmit: use the ref from layers (shared with tool's ToolEmitTag service)
      const toolEmitRef = toolEmitRefs.get(forkId)!


      // Build tool tagName → defKey lookup and defKey → tool lookup
      const tagToDefKey = new Map<string, string>()
      const tagToTool = new Map<string, Tool.Any>()
      for (const [defKey, tool] of Object.entries(agentDef.tools)) {
        if (!tool) continue
        const t = tool as Tool.Any
        const tagName = defaultXmlTagName(t)
        tagToDefKey.set(tagName, defKey)
        tagToTool.set(tagName, t)
      }

      /** Resolve a xml-act event's tagName to the definition key. */
      const resolveKey = (tagName: string): string => {
        return tagToDefKey.get(tagName) ?? tagName
      }

      // Also build callable-based lookup for resolveKey from group+toolName
      const callableToKey = new Map<string, string>()
      for (const [key, tool] of Object.entries(agentDef.tools)) {
        const t = tool as { name: string; group?: string }
        const group = t.group
        const callable = (group && group !== 'default') ? `${group}.${t.name}` : t.name
        callableToKey.set(callable, key)
      }
      const resolveKeyFromCallable = (group: string, toolName: string): string => {
        const callable = group === 'default' ? toolName : `${group}.${toolName}`
        return callableToKey.get(callable) ?? callable
      }

      // Track tools called (by definition key) for turn policy
      const toolsCalledKeys: string[] = []
      let lastToolKey: string | null = null
      const toolCalls: TurnToolCall[] = []
      const messagesSent: Array<{ id: string, dest: string }> = []
      let hasAnyMessage = false

      // Track tool input (ToolInputReady provides the parsed input)
      const toolInputs = new Map<string, unknown>()

      // Track cached tool calls (replay) — skip their events
      const cachedToolCallIds = new Set<string>()

      // Position counter for tool events
      let positionCounter = 0

      // Track tag names for ToolStarted events
      const toolCallTagNames = new Map<string, string>()

      // Accumulate inspect block results
      const inspectResults: InspectResult[] = []

      // Track successful tool refs for synthetic inspect injection (in execution order)
      const toolRefs: { tag: string; tree: OutputNode }[] = []

      let hasStructuralParseError = false
      let structuralParseErrorReminder: string | null = null

      // Track messages to nonexistent agent IDs
      let hasNonexistentAgentDest = false
      let nonexistentAgentError: string | null = null

      // Accumulate raw XML
      let accumulatedCode = ''

      // Store execution result
      let executionResult: TurnResult = { success: true, turnDecision: 'yield' }

      // Create the PolicyContextProvider for turn policy evaluation
      const cwd = forkCwds.get(forkId) ?? process.cwd()
      const policyCtxProvider = createPolicyContextProvider(
        forkId,
        cwd,
        yield* AgentRegistryProjection.Tag,
        yield* WorkingStateProjection.Tag,

      )

      // Run xml-act runtime
      const eventStream = runtime.streamWith(xmlStream, { initialState: replayState })

      // Track toolCallId → toolKey mapping for event forwarding
      const toolCallKeys = new Map<string, string>()

      yield* Effect.scoped(
        eventStream.pipe(
          Stream.provideLayer(layers),
          Stream.runForEach((event: XmlRuntimeEvent) => Effect.gen(function* () {
            switch (event._tag) {
              // --- Tool Input Started ---
              case 'ToolInputStarted': {
                const toolKey = resolveKeyFromCallable(event.group, event.toolName)
                toolCallTagNames.set(event.toolCallId, event.tagName)
                toolCallKeys.set(event.toolCallId, toolKey)

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
                // Reset tool emit ref for the new tool execution
                yield* Ref.set(toolEmitRef, undefined)

                const toolKey = toolCallKeys.get(event.toolCallId) ?? resolveKeyFromCallable(event.group, event.toolName)
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

                // Skip cached tool calls — these are replays
                if (cachedToolCallIds.has(event.toolCallId)) {
                  cachedToolCallIds.delete(event.toolCallId)
                  break
                }

                // Track tool calls for turn policy
                toolsCalledKeys.push(toolKey)
                lastToolKey = toolKey

                toolInputs.delete(event.toolCallId)

                // Read and clear tool emit
                const emittedValue = yield* Ref.getAndSet(toolEmitRef, undefined)

                // Map XmlToolResult → ToolResult for TurnToolCall accumulation
                const toolResult: ToolResult = mapXmlToolResult(event.result, emittedValue)

                toolCalls.push({
                  toolKey,
                  group: event.group,
                  toolName: event.toolName,
                  result: toolResult,
                })

                if (event.result._tag === 'Success') {
                  toolRefs.push({ tag: event.result.ref.tag, tree: event.result.ref.tree })
                }

                // Forward event (attach display data if emitted by tool)
                yield* Queue.offer(sink, {
                  _tag: 'ToolEvent',
                  toolCallId: event.toolCallId,
                  toolKey,
                  event,
                  ...(emittedValue ? { display: emittedValue } : {}),
                })
                break
              }

              // --- Tool streaming events (field values, body chunks, children) ---
              case 'ToolInputBodyChunk':
              case 'ToolInputFieldValue':
              case 'ToolInputChildStarted':
              case 'ToolInputChildComplete': {
                const toolKey = toolCallKeys.get(event.toolCallId) ?? event.toolCallId
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
                messagesSent.push({ id: event.id, dest: event.dest })
                hasAnyMessage = true

                // Validate agent message destinations inline during execution
                if (event.dest !== 'user' && event.dest !== 'parent') {
                  const currentForkState = yield* forkProjectionInst.get
                  const targetFork = [...currentForkState.forks.values()].find(f => f.agentId === event.dest && f.status === 'running')
                  if (!targetFork) {
                    hasNonexistentAgentDest = true
                    const destStr = `"${event.dest}"`
                    nonexistentAgentError = `<error>\n${formatNonexistentAgentError(destStr)}\n</error>`
                  }
                }

                yield* Queue.offer(sink, { _tag: 'MessageStart', id: event.id, dest: event.dest })
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


              // --- Inspect Resolved (ref resolved in inspect block) ---
              case 'InspectResolved': {
                inspectResults.push({ status: 'resolved', toolRef: event.toolRef, query: event.query, content: event.content })
                break
              }

              // --- Invalid Ref (tool ref doesn't exist) ---
              case 'InvalidRef': {
                inspectResults.push({ status: 'invalid_ref', toolRef: event.toolRef })
                break
              }

              case 'StructuralParseError': {
                hasStructuralParseError = true
                const msg = event.error._tag === 'UnclosedThink'
                  ? UNCLOSED_THINK_REMINDER
                  : event.error._tag === 'UnclosedActions'
                    ? UNCLOSED_ACTIONS_REMINDER
                    : UNCLOSED_INSPECT_REMINDER
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
                  const hasInvalidRefs = inspectResults.some(ir => ir.status === 'invalid_ref')

                  if (hasToolErrors || hasInvalidRefs || hasStructuralParseError || hasNonexistentAgentDest) {
                    executionResult = {
                      success: true,
                      turnDecision: 'continue',
                      ...(structuralParseErrorReminder
                        ? { reminder: structuralParseErrorReminder }
                        : nonexistentAgentError
                          ? { reminder: nonexistentAgentError }
                          : {}),
                    }
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
                    executionResult = { success: true, turnDecision: turnResult.action as TurnDecision, reminder: turnResult.reminder }
                  }
                } else if (endResult._tag === 'Interrupted') {
                  executionResult = { success: false, error: 'Interrupted', cancelled: true }
                } else if (endResult._tag === 'Failure') {
                  executionResult = { success: false, error: endResult.error, cancelled: false }
                } else if (endResult._tag === 'GateRejected') {
                  const rejection = endResult.rejection
                  const isPermissionRejection = rejection && typeof rejection === 'object' && '_tag' in rejection
                  if (isPermissionRejection) {
                    const r = rejection as { _tag: string; reason: string }
                    const cancelled = r._tag === 'UserRejection'
                    executionResult = { success: false, error: r.reason || 'Gate rejected', cancelled }
                  } else {
                    executionResult = { success: false, error: String(rejection) || 'Gate rejected', cancelled: true }
                  }
                }
                break
              }
            }
          }))
        )
      )

      let syntheticInspectCode: string | undefined
      if (inspectResults.length === 0 && toolRefs.length > 0) {
        const totals = new Map<string, number>()
        for (const ref of toolRefs) {
          totals.set(ref.tag, (totals.get(ref.tag) ?? 0) + 1)
        }
        const seen = new Map<string, number>()

        const inspectLines: string[] = []
        inspectLines.push('<inspect>')
        for (const ref of toolRefs) {
          const index = (seen.get(ref.tag) ?? 0) + 1 // 1-based index within this tag
          seen.set(ref.tag, index)
          const total = totals.get(ref.tag) ?? index
          const recency = total - index
          const toolRef = recency === 0 ? ref.tag : `${ref.tag}~${recency}`
          inspectLines.push(`<ref tool="${toolRef}" />`)
          inspectResults.push({ status: 'resolved', toolRef, content: outputToText(ref.tree) })
        }
        inspectLines.push('</inspect>')
        syntheticInspectCode = '\n' + inspectLines.join('\n') + '\n'
      }

      return {
        result: executionResult,
        code: accumulatedCode,
        toolCalls,
        inspectResults,
        syntheticInspectCode,
      }
    }),

    initFork: (forkId, variant) => Effect.gen(function* () {

      const sessionContextProjection = yield* SessionContextProjection.Tag
      const forkProjection = yield* ForkProjection.Tag
      const artifactProjection = yield* ArtifactProjection.Tag
      const agentRegistryProjection = yield* AgentRegistryProjection.Tag
      const workingStateProjection = yield* WorkingStateProjection.Tag

      const conversationProjection = yield* ConversationProjection.Tag
      const persistence = yield* ChatPersistence
      const persistenceLayer = Layer.succeed(ChatPersistence, persistence)

      // Create ToolEmit ref (shared with execute() event handler via toolEmitRefs map)
      const toolEmitRef = yield* Ref.make<ToolDisplay | undefined>(undefined)
      toolEmitRefs.set(forkId, toolEmitRef)

      const sessionState = yield* sessionContextProjection.get
      const cwd = sessionState.context?.cwd ?? process.cwd()

      let layers = makeForkLayers(
        forkId,
        sessionContextProjection, forkProjection,
        artifactProjection, agentRegistryProjection, workingStateProjection,
        conversationProjection,
        blockingDeferreds, approvalState,
        persistenceLayer, permissionInterceptor, toolEmitRef, cwd,
      )
      forkCwds.set(forkId, cwd)

      // Inject browser harness for browser agent forks
      if (variant === 'browser' && forkId) {
        const browserService = yield* BrowserService
        const harness = yield* browserService.get(forkId)
        layers = Layer.merge(layers, Layer.succeed(BrowserHarnessTag, harness))
      }

      // Store variant for agent resolution
      if (forkId !== null) {
        forkAgentVariants.set(forkId, variant)
      }

      const projectionReader: ProjectionReader = {
        getAgentRegistry: () => agentRegistryProjection.get,
      }
      const projectionReaderLayer = Layer.succeed(ProjectionReaderTag, projectionReader)
      layers = Layer.merge(layers, projectionReaderLayer)

      // Cache the layers
      forkLayers.set(forkId, layers)

      // Bind observables
      const agentDef = getAgentDefinition(variant)
      const agentObservables = agentDef.observables.map(obs =>
        bindObservable(obs, () => Effect.succeed(layers))
      )
      boundObservables.set(forkId, agentObservables)
    }),

    disposeFork: (forkId) => Effect.gen(function* () {
      forkLayers.delete(forkId)
      forkCwds.delete(forkId)
      boundObservables.delete(forkId)
    }),

    fork: (params) => Effect.gen(function* () {
      const forkId = createId()
      forkAgentVariants.set(forkId, params.role)
      const workerBus = yield* WorkerBusTag<AppEvent>()
      const context = yield* buildForkContext(params)

      yield* service.initFork(forkId, params.role)

      yield* workerBus.publish({
        type: 'fork_started',
        forkId,
        parentForkId: params.parentForkId,
        name: params.name,
        agentId: params.agentId,
        context,
        outputSchema: params.outputSchema,
        blocking: false,
        mode: params.mode,
        role: params.role,
        taskId: params.taskId,
      })

      return forkId
    }),

    forkSync: (params) => Effect.gen(function* () {
      const forkId = createId()

      const deferred = yield* Deferred.make<unknown, never>()
      blockingDeferreds.set(forkId, deferred)

      const workerBus = yield* WorkerBusTag<AppEvent>()
      const augmentedPrompt = params.timeLimit
        ? `${params.prompt}\n\n[Time budget: ~${Math.round(params.timeLimit / 1000)}s. Use report() to log findings as you go.]`
        : params.prompt
      const context = yield* buildForkContext({ ...params, prompt: augmentedPrompt })

      forkAgentVariants.set(forkId, params.role)
      yield* service.initFork(forkId, params.role)

      yield* workerBus.publish({
        type: 'fork_started',
        forkId,
        parentForkId: params.parentForkId,
        name: params.name,
        agentId: params.agentId,
        context,
        outputSchema: params.outputSchema,
        blocking: true,
        mode: params.mode,
        role: params.role,
        taskId: params.taskId,
      })

      if (params.timeLimit) {
        const result = yield* Effect.raceFirst(
          Deferred.await(deferred).pipe(Effect.map(r => ({ _tag: 'completed' as const, value: r }))),
          Effect.sleep(params.timeLimit).pipe(Effect.map(() => ({ _tag: 'timeout' as const })))
        )

        if (result._tag === 'timeout') {
          blockingDeferreds.delete(forkId)
          yield* workerBus.publish({ type: 'interrupt', forkId } as AppEvent)
          return '[Research timed out]'
        }

        return result.value
      }

      const result = yield* Deferred.await(deferred)
      return result
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
