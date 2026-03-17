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
import { WorkingStateProjection, isStable } from './projections/working-state'
import { TurnProjection } from './projections/turn'
import { CanonicalTurnProjection } from './projections/canonical-turn'
import { MemoryProjection } from './projections/memory'
import { SubagentActivityProjection } from './projections/subagent-activity'
import { DisplayProjection } from './projections/display'
import { AgentRoutingProjection } from './projections/agent-routing'
import { AgentStatusProjection } from './projections/agent-status'
import { CompactionProjection } from './projections/compaction'

import { ArtifactProjection } from './projections/artifact'

import { ReplayProjection } from './projections/replay'
import { ChatTitleProjection } from './projections/chat-title'
import { ConversationProjection } from './projections/conversation'
import { UserPresenceProjection } from './projections/user-presence'
import { OutboundMessagesProjection } from './projections/outbound-messages'
import { ArtifactAwarenessProjection } from './projections/artifact-awareness'

// Workers
import { TurnController } from './workers/turn-controller'
import { Cortex } from './workers/cortex'
import { AgentOrchestrator } from './workers/agent-orchestrator'
import { LifecycleCoordinator } from './workers/lifecycle-coordinator'

import { Autopilot } from './workers/autopilot'
import { CompactionWorker } from './workers/compaction-worker'
import { ApprovalWorker } from './workers/approval-worker'
import type { AgentVariant } from './agents'
import { ChatTitleWorker } from './workers/chat-title-worker'
import { UserPresenceWorker } from './workers/user-presence-worker'
import { ArtifactSyncWorker } from './workers/artifact-sync-worker'
import { FileMentionResolver } from './workers/file-mention-resolver'

// Execution
import { ExecutionManager, ExecutionManagerLive } from './execution/execution-manager'
import { BrowserServiceLive } from './services/browser-service'
import { registerApprovalBridge } from './execution/approval-bridge'

// Persistence
import { ChatPersistence } from './persistence/chat-persistence-service'

// Utils
import { collectSessionContext } from './util/collect-session-context'

// Providers
import { bootstrapProviderRuntime, makeModelResolver, makeNoopTracer, makeProviderRuntimeLive, makeTracePersister } from '@magnitudedev/providers'
import type { StorageClient } from '@magnitudedev/storage'
import { initLogger } from '@magnitudedev/logger'
import { writeTrace, initTraceSession } from '@magnitudedev/tracing'
import { createMemoryExtractionJob, drainPendingJobsOnStartup, spawnDetachedMemoryExtractionWorker, writePendingJob } from './memory/job-queue'
import { EphemeralSessionContextTag } from './agents/types'

// =============================================================================
// Agent
// =============================================================================

export const CodingAgent = Agent.define<AppEvent>()({
  name: 'CodingAgent',

  projections: [
    SessionContextProjection,
    AgentRoutingProjection,
    AgentStatusProjection,
    CompactionProjection,
    WorkingStateProjection,
    TurnProjection,
    CanonicalTurnProjection,

    ArtifactProjection,

    ReplayProjection,
    SubagentActivityProjection,
    OutboundMessagesProjection,
    ArtifactAwarenessProjection,
    MemoryProjection,
    DisplayProjection,
    ChatTitleProjection,
    ConversationProjection,
    UserPresenceProjection,
  ],

  workers: [
    TurnController,
    Cortex,
    AgentOrchestrator,
    LifecycleCoordinator,
    Autopilot,
    CompactionWorker,
    ApprovalWorker,

    ArtifactSyncWorker,
    FileMentionResolver,

    ChatTitleWorker,
    UserPresenceWorker,
  ],

  expose: {
    signals: {
      restoreQueuedMessages: DisplayProjection.signals.restoreQueuedMessages,

      chatTitleGenerated: ChatTitleProjection.signals.chatTitleGenerated
    },
    state: {
      display: DisplayProjection,
      working: WorkingStateProjection,
      memory: MemoryProjection,
      compaction: CompactionProjection,
      agentRouting: AgentRoutingProjection,
      agentStatus: AgentStatusProjection,

      artifacts: ArtifactProjection,
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
  storage: StorageClient

  /**
   * Enable LLM call tracing to ~/.magnitude/traces/
   */
  debug?: boolean

  /**
   * Provide a pre-built session context instead of collecting from the local environment.
   * Useful for evals / headless runs where the agent operates in a container.
   */
  sessionContext?: SessionContext

  /**
   * Optional pre-configured provider runtime.
   * When provided, provider bootstrap is skipped and the caller is responsible
   * for initializing model selections/auth inside the runtime.
   */
  providerRuntime?: ReturnType<typeof makeProviderRuntimeLive>

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
let hasDrainedPendingMemoryJobs = false

export async function createCodingAgentClient(options: CreateClientOptions) {
  const memoryEnabled = await options.storage.config.getMemoryEnabled()

  // Bootstrap provider runtime from stored config / env vars unless the caller
  // supplied a pre-configured runtime (e.g. headless oneshot mode).
  const providerRuntime = options.providerRuntime ?? makeProviderRuntimeLive()
  if (!options.providerRuntime) {
    await Effect.runPromise(bootstrapProviderRuntime.pipe(Effect.provide(providerRuntime)))
  }

  if (!hasDrainedPendingMemoryJobs) {
    hasDrainedPendingMemoryJobs = true
    if (memoryEnabled) {
      drainPendingJobsOnStartup(options.storage).catch(() => {})
    }
  }

  // Enable tracing in debug mode
  if (options.debug) {
    const traceSessionId = new Date().toISOString().replace(/:/g, '-').replace(/\.\d{3}Z$/, 'Z')
    initTraceSession(traceSessionId, { cwd: process.cwd(), platform: process.platform, gitBranch: null })

  }

  const tracerLayer = options.debug ? makeTracePersister((trace) => writeTrace(trace)) : makeNoopTracer()
  const ephemeralSessionContextLayer = Layer.succeed(EphemeralSessionContextTag, {
    disableShellSafeguards: options.disableShellSafeguards ?? false,
    disableCwdSafeguards: options.disableCwdSafeguards ?? false,
  })
  const layer = Layer.mergeAll(
    Layer.provide(ExecutionManagerLive, ephemeralSessionContextLayer),
    Layer.provide(BrowserServiceLive, providerRuntime),
    Layer.provide(makeModelResolver(), providerRuntime),
    providerRuntime,
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

    // Bridge approval state into display and working-state projections
    yield* registerApprovalBridge

    const events = yield* persistence.loadEvents()

    if (events.length === 0) {
      // New session
      const context = options.sessionContext ?? (yield* Effect.promise(() => collectSessionContext({
        cwd: process.cwd(),
        memoryEnabled,
        storage: options.storage,
      })))

      yield* Effect.promise(() => client.send({
        type: 'session_initialized',
        forkId: null,
        context
      }))

      // Persist the initial event immediately
      const pending = yield* eventSink.drainPending()
      if (pending.length > 0) {
        yield* persistence.persistNewEvents(pending)
      }
    } else {
      // Existing session — hydrate
      yield* hydrationContext.setHydrating(true)

      for (const event of events) {
        yield* Effect.promise(() => client.send(event))
      }

      yield* Effect.sleep('50 millis')
      yield* hydrationContext.setHydrating(false)

      const executionManager = yield* ExecutionManager
      const agentStatusProjection = yield* AgentStatusProjection.Tag
      const workingStateProjection = yield* WorkingStateProjection.Tag

      // Create root sandbox (hydration happens lazily in execute())
      yield* executionManager.initFork(null, 'orchestrator')

      // Create execution resources for all non-dismissed agents
      const agentState = yield* agentStatusProjection.get
      for (const [, agent] of agentState.agents) {
        if (agent.status === 'dismissed') continue
        yield* executionManager.initFork(agent.forkId, agent.role as AgentVariant)
      }

      // Hydration recovery: detect agents that were in-flight when the process
      // died. After replay, WorkingState tells us whether each agent fork is in
      // a valid settled state. If not stable, emit an interrupt to cleanly
      // terminate it through the normal recovery chain.
      for (const [, agent] of agentState.agents) {
        if (agent.status === 'dismissed') continue
        const forkWorkingState = yield* workingStateProjection.getFork(agent.forkId)
        if (!isStable(forkWorkingState)) {
          yield* Effect.promise(() => client.send({
            type: 'interrupt',
            forkId: agent.forkId,
          }))
        }
      }

      // Same check for the root fork (orchestrator, forkId=null).
      // Root doesn't get fork_completed, but the interrupt still closes
      // any in-flight turn via the turnInterrupted → turn_completed path.
      const rootState = yield* workingStateProjection.getFork(null)
      if (!isStable(rootState)) {
        yield* Effect.promise(() => client.send({
          type: 'interrupt',
          forkId: null,
        }))
      }

      // NOTE: AgentStatusProjection is the source of truth for agent identity, metadata, and execution state.
      // AgentRoutingProjection handles message routing only. forkId remains the execution handle used by forked projections/workers.

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

  // NOTE: Memory extraction trigger lives only here (wrapped dispose) so all teardown paths
  // (CLI and server) converge through one durable marker + best-effort detached worker flow.
  const dispose = async () => {

    let jobPath: string | null = null
    if (memoryEnabled) {
      try {
        const metadata = await client.runEffect(Effect.gen(function* () {
          const persistence = yield* ChatPersistence
          return yield* persistence.getSessionMetadata()
        }))
        const cwd = metadata.workingDirectory
        const sessionId = metadata.sessionId
        const eventsPath = options.storage.sessions.getEventsPath(sessionId)
        const memoryPath = options.storage.memory.getPath()
        const job = createMemoryExtractionJob({ sessionId, cwd, eventsPath, memoryPath })
        jobPath = await writePendingJob(options.storage, job)
      } catch {}
    }

    try {
      // Best-effort flush of pending events to disk. If the session was mid-turn,
      // hydration recovery will detect non-stable forks on next startup and emit
      // interrupts to bring them to a clean terminal state.
      await client.runEffect(flushPendingEvents())
    } catch {}

    await originalDispose()

    if (jobPath) {
      spawnDetachedMemoryExtractionWorker(jobPath)
    }
  }

  return {
    ...client,
    dispose,
    subscribeDebug
  }
}


