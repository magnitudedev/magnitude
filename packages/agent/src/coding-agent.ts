/**
 * Coding Agent
 *
 * A minimal coding agent that:
 * - Uses event-core architecture (projections, workers, signals)
 * - Uses js-act for sandbox execution with tool calling
 * - Has a shell command tool for executing commands
 * - Supports session persistence and hydration
 */

import { Effect, Layer, Stream } from 'effect'
import { Agent } from '@magnitudedev/event-core'
import { HydrationContext, EventSinkTag } from '@magnitudedev/event-core'
import type { AppEvent, SessionContext } from './events'
import type { DebugSnapshot } from './projections/debug-introspection'

// Projections
import { SessionContextProjection } from './projections/session-context'
import { TurnProjection } from './projections/turn'
import { CanonicalTurnProjection } from './projections/canonical-turn'
import { MemoryProjection } from './projections/memory'
import { SubagentActivityProjection } from './projections/subagent-activity'
import { DisplayProjection } from './projections/display'
import { ToolStateProjection } from './projections/tool-state'
import { AgentRoutingProjection } from './projections/agent-routing'
import { AgentStatusProjection } from './projections/agent-status'
import { TaskGraphProjection } from './projections/task-graph'
import { TaskWorkerProjection } from './projections/task-worker'
import { CompactionProjection } from './projections/compaction'
import { ReplayProjection } from './projections/replay'
import { ConversationProjection } from './projections/conversation'
import { UserPresenceProjection } from './projections/user-presence'
import { OutboundMessagesProjection } from './projections/outbound-messages'
import { UserMessageResolutionProjection } from './projections/user-message-resolution'


// Workers
import { TurnController } from './workers/turn-controller'
import { Cortex } from './workers/cortex'
import { AgentLifecycle } from './workers/agent-lifecycle'
import { LifecycleCoordinator } from './workers/lifecycle-coordinator'

import { Autopilot } from './workers/autopilot'
import { CompactionWorker } from './workers/compaction-worker'
import { ApprovalWorker } from './workers/approval-worker'
import { isValidVariant, type AgentVariant } from './agents'
import { UserPresenceWorker } from './workers/user-presence-worker'
import { FileMentionResolver } from './workers/file-mention-resolver'
import { SessionTitleWorker } from './workers/session-title-worker'
import { FsLive } from './services/fs'

// Execution
import { ExecutionManager, ExecutionManagerLive } from './execution/execution-manager'
import { BrowserServiceLive } from './services/browser-service'
import { registerApprovalBridge } from './execution/approval-bridge'

// Persistence
import { ChatPersistence } from './persistence/chat-persistence-service'

// Utils
import { collectSessionContext } from './util/collect-session-context'

// Providers
import { bootstrapProviderRuntime, makeModelResolver, makeNoopTracer, makeProviderRuntimeLive, makeTracePersister, type ProviderRuntime } from '@magnitudedev/providers'
import { MAGNITUDE_SLOTS, type MagnitudeSlot } from './model-slots'
import type { StorageClient } from '@magnitudedev/storage'
import { initLogger, logger } from '@magnitudedev/logger'
import { writeTrace, initTraceSession } from '@magnitudedev/tracing'

import { EphemeralSessionContextTag } from './agents/types'
import { publishConfigFromProviders } from './ambient/config-ambient'
import { loadSkills } from '@magnitudedev/skills'
import { SkillsAmbient, publishSkills } from './ambient/skills-ambient'


// =============================================================================
// Agent
// =============================================================================

export const CodingAgent = Agent.define<AppEvent>()({
  name: 'CodingAgent',

  projections: [
    SessionContextProjection,
    AgentRoutingProjection,
    AgentStatusProjection,
    TaskGraphProjection,
    CompactionProjection,
    TurnProjection,
    CanonicalTurnProjection,

    ReplayProjection,
    SubagentActivityProjection,
    OutboundMessagesProjection,
    UserMessageResolutionProjection,
    ToolStateProjection,
    MemoryProjection,
    TaskWorkerProjection,
    DisplayProjection,
    ConversationProjection,
    UserPresenceProjection,
  ],

  workers: [
    TurnController,
    Cortex,
    AgentLifecycle,
    LifecycleCoordinator,
    Autopilot,
    CompactionWorker,
    ApprovalWorker,

    FileMentionResolver,

    UserPresenceWorker,
    SessionTitleWorker,
  ],

  expose: {
    signals: {
      restoreQueuedMessages: DisplayProjection.signals.restoreQueuedMessages,

      taskCreated: TaskGraphProjection.signals.taskCreated,
      taskCompleted: TaskGraphProjection.signals.taskCompleted,
      taskCancelled: TaskGraphProjection.signals.taskCancelled,
      taskStatusChanged: TaskGraphProjection.signals.taskStatusChanged
    },
    state: {
      display: DisplayProjection,
      toolState: ToolStateProjection,
      turn: TurnProjection,
      memory: MemoryProjection,
      compaction: CompactionProjection,
      agentRouting: AgentRoutingProjection,
      agentStatus: AgentStatusProjection,
      taskGraph: TaskGraphProjection,
      taskWorker: TaskWorkerProjection,
    }
  }
})

// =============================================================================
// Client Factory
// =============================================================================

export interface CreateClientOptions {
  /**
   * Persistence service for session storage and hydration.
   */
  persistence: Layer.Layer<ChatPersistence, never, never>

  /**
   * Storage client for config, sessions, memory, and memory jobs.
   */
  storage: StorageClient<MagnitudeSlot>

  /**
   * Enable LLM call tracing to ~/.magnitude/traces/
   */
  debug?: boolean

  /**
   * Provide a pre-built session context instead of collecting from the local environment.
   * Useful for evals / headless runs where the agent operates in a container.
   */
  sessionContext?: Omit<SessionContext, 'workspacePath'>

  /**
   * Optional pre-configured provider runtime.
   * When provided, provider bootstrap is skipped and the caller is responsible
   * for initializing model selections/auth inside the runtime.
   */
  providerRuntime?: ProviderRuntime<MagnitudeSlot>

  /**
   * Disable shell command classification safeguards for this runtime only.
   */
  disableShellSafeguards?: boolean

  /**
   * Disable working-directory boundary safeguards for this runtime only.
   */
  disableCwdSafeguards?: boolean
}

/**
 * Create a CodingAgent client with persistence.
 *
 * Loads events from persistence on startup:
 * - If events exist: hydrates projections and sandbox from persisted state
 * - If no events: initializes a new session
 */
export async function createCodingAgentClient(options: CreateClientOptions) {

  // Bootstrap provider runtime from stored config / env vars unless the caller
  // supplied a pre-configured runtime (e.g. headless oneshot mode).
  const providerRuntime = options.providerRuntime ?? makeProviderRuntimeLive<MagnitudeSlot>()
  if (!options.providerRuntime) {
    await Effect.runPromise(bootstrapProviderRuntime<MagnitudeSlot>({ slots: MAGNITUDE_SLOTS }).pipe(Effect.provide(providerRuntime)))
  }

  // Enable tracing in debug mode
  if (options.debug) {
    const traceSessionId = new Date().toISOString().replace(/:/g, '-').replace(/\.\d{3}Z$/, 'Z')
    initTraceSession(traceSessionId, { cwd: process.cwd(), platform: process.platform, gitBranch: null })

  }

  const tracerLayer = options.debug ? makeTracePersister((trace) => writeTrace(trace as any)) : makeNoopTracer()
  const ephemeralSessionContextLayer = Layer.succeed(EphemeralSessionContextTag, {
    disableShellSafeguards: options.disableShellSafeguards ?? false,
    disableCwdSafeguards: options.disableCwdSafeguards ?? false,
  })
  const layer = Layer.mergeAll(
    Layer.provide(ExecutionManagerLive, ephemeralSessionContextLayer),
    Layer.provide(BrowserServiceLive, providerRuntime),
    Layer.provide(makeModelResolver<MagnitudeSlot>(), providerRuntime),
    providerRuntime,
    FsLive,
    tracerLayer,
    options.persistence,
  )
  const client = await CodingAgent.createClient(layer)

  try {
    const metadata = await client.runEffect(Effect.gen(function* () {
      const persistence = yield* ChatPersistence
      return yield* persistence.getSessionMetadata()
    }))
    initLogger(metadata.sessionId)
  } catch {}

  const flushPendingEvents = () => Effect.gen(function* () {
    const persistence = yield* ChatPersistence
    const eventSink = yield* EventSinkTag<AppEvent>()
    const pending = yield* eventSink.drainPending()
    if (pending.length > 0) {
      yield* persistence.persistNewEvents(pending)
    }
  })

  await client.runEffect(Effect.gen(function* () {
    const persistence = yield* ChatPersistence
    const hydrationContext = yield* HydrationContext
    const eventSink = yield* EventSinkTag<AppEvent>()

    // Bridge approval state into display and turn projections
    yield* registerApprovalBridge

    const events = yield* persistence.loadEvents()

    if (events.length === 0) {
      // New session
      const baseContext = options.sessionContext ?? (yield* Effect.tryPromise(async () => {
        try {
          return await collectSessionContext({
            cwd: process.cwd(),
            storage: options.storage,
          })
        } catch (err) {
          logger.error({ err }, 'Failed to collect session context')
          // Should not happen, but return minimal context so session can still initialize
          const cwd = process.cwd()
          return {
            cwd,
            platform: process.platform === 'darwin' ? 'macos' as const : process.platform === 'win32' ? 'windows' as const : 'linux' as const,
            shell: process.env.SHELL?.split('/').pop() || 'bash',
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            username: process.env.USER || 'unknown',
            fullName: null,
            git: null,
            folderStructure: '(failed to collect folder structure)',
            agentsFile: null,
            skills: null,
          }
        }
      }))

      const sessionMetadata = yield* persistence.getSessionMetadata()
      const workspacePath = yield* Effect.promise(() =>
        options.storage.sessions.createWorkspace(sessionMetadata.sessionId, baseContext.cwd)
      )
      const context: SessionContext = {
        ...baseContext,
        workspacePath,
      }

      yield* Effect.promise(() => client.send({
        type: 'session_initialized',
        forkId: null,
        context
      }))

      // Load skills from standard directories
      const skills = yield* Effect.tryPromise(() => loadSkills(process.cwd()))
      yield* publishSkills(skills)

      // Persist the initial event immediately
      const pending = yield* eventSink.drainPending()
      if (pending.length > 0) {
        yield* persistence.persistNewEvents(pending)
      }

    } else {
      // Existing session — hydrate
      // Ensure workspace exists and symlink is up-to-date
      const sessionMetadata = yield* persistence.getSessionMetadata()
      yield* Effect.promise(() =>
        options.storage.sessions.createWorkspace(sessionMetadata.sessionId, process.cwd())
      )

      yield* hydrationContext.setHydrating(true)

      for (const event of events) {
        yield* Effect.promise(() => client.send(event))
      }

      yield* Effect.sleep('50 millis')
      yield* hydrationContext.setHydrating(false)

      const executionManager = yield* ExecutionManager
      const agentStatusProjection = yield* AgentStatusProjection.Tag
      const turnProjection = yield* TurnProjection.Tag

      // Create root sandbox (hydration happens lazily in execute())
      const sessionContextState = yield* (yield* SessionContextProjection.Tag).get
      const rootVariant = sessionContextState.context?.oneshot ? 'lead-oneshot' : 'lead'
      yield* executionManager.initFork(null, rootVariant)

      // Create execution resources for all known agents.
      const agentState = yield* agentStatusProjection.get
      for (const [, agent] of agentState.agents) {
        if (!isValidVariant(agent.role)) {
          continue
        }
        yield* executionManager.initFork(agent.forkId, agent.role)
      }

      // Hydration recovery: detect forks that were in-flight when the process
      // died. If a fork is still active/interrupting after replay, synthesize
      // a cancelled terminal event to close turn lifecycle deterministically.
      for (const [, agent] of agentState.agents) {
        const forkTurnState = yield* turnProjection.getFork(agent.forkId)
        if (forkTurnState._tag === 'active' || forkTurnState._tag === 'interrupting') {
          yield* Effect.promise(() => client.send({
            type: 'turn_completed',
            forkId: agent.forkId,
            turnId: forkTurnState.turnId,
            chainId: forkTurnState.chainId,
            strategyId: 'xml-act',

            result: {
              success: false,
              error: 'Hydration recovery: cancelled in-flight turn',
              cancelled: true,
            },
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
            providerId: null,
            modelId: null,
          }))
        }
      }

      // Same recovery for root fork.
      const rootTurnState = yield* turnProjection.getFork(null)
      if (rootTurnState._tag === 'active' || rootTurnState._tag === 'interrupting') {
        yield* Effect.promise(() => client.send({
          type: 'turn_completed',
          forkId: null,
          turnId: rootTurnState.turnId,
          chainId: rootTurnState.chainId,
          strategyId: 'xml-act',
          result: {
            success: false,
            error: 'Hydration recovery: cancelled in-flight turn',
            cancelled: true,
          },
          inputTokens: null,
          outputTokens: null,
          cacheReadTokens: null,
          cacheWriteTokens: null,
          providerId: null,
          modelId: null,
        }))
      }

      // NOTE: AgentStatusProjection is the source of truth for agent identity, metadata, and execution state.
      // AgentRoutingProjection handles message routing only. forkId remains the execution handle used by forked projections/workers.

      // Load skills from standard directories
      const skills = yield* Effect.tryPromise(() => loadSkills(process.cwd()))
      yield* publishSkills(skills)

      // Persist all recovery events immediately so reopening the same session
      // again won't re-run recovery for already-terminated forks.
      yield* flushPendingEvents()
    }
  }))

  // Debug subscription support
  const subscribeDebug = (forkId: string | null, callback: (snapshot: DebugSnapshot) => void): (() => void) => {
    let isActive = true

    const effect = Effect.gen(function* () {
      const { createDebugStream } = yield* Effect.promise(() => import('./projections/debug-introspection'))
      const stream = yield* createDebugStream(forkId)

      // Emit initial snapshot
      const { getDebugSnapshot } = yield* Effect.promise(() => import('./projections/debug-introspection'))
      const initial = yield* getDebugSnapshot(forkId)
      if (isActive) callback(initial)

      // Subscribe to stream
      yield* stream.pipe(
        Stream.takeWhile(() => isActive),
        Stream.runForEach((snapshot) =>
          Effect.sync(() => {
            if (isActive) callback(snapshot)
          })
        )
      )
    })

    client.runEffect(effect)

    return () => { isActive = false }
  }

  const originalDispose = client.dispose.bind(client)

  const dispose = async () => {
    try {
      // Best-effort flush of pending events to disk. If the session was mid-turn,
      // hydration recovery will detect non-stable forks on next startup and emit
      // interrupts to bring them to a clean terminal state.
      await client.runEffect(flushPendingEvents())
    } catch {}

    await originalDispose()
  }

  const refreshConfig = () => client.runEffect(publishConfigFromProviders)

  return {
    ...client,
    dispose,
    subscribeDebug,
    refreshConfig,
  }
}


