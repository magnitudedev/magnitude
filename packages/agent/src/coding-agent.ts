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
import { WorkingStateProjection } from './projections/working-state'
import { TurnProjection } from './projections/turn'
import { CanonicalTurnProjection } from './projections/canonical-turn'
import { MemoryProjection } from './projections/memory'
import { SubagentActivityProjection } from './projections/subagent-activity'
import { DisplayProjection } from './projections/display'
import { ForkProjection } from './projections/fork'
import { CompactionProjection } from './projections/compaction'

import { ArtifactProjection } from './projections/artifact'
import { AgentRegistryProjection } from './projections/agent-registry'

import { ReplayProjection } from './projections/replay'
import { ChatTitleProjection } from './projections/chat-title'
import { ConversationProjection } from './projections/conversation'
import { UserPresenceProjection } from './projections/user-presence'
import { OutboundMessagesProjection } from './projections/outbound-messages'
import { ArtifactAwarenessProjection } from './projections/artifact-awareness'

// Workers
import { TurnController } from './workers/turn-controller'
import { Cortex } from './workers/cortex'
import { ForkOrchestrator } from './workers/fork-orchestrator'
import { LifecycleCoordinator } from './workers/lifecycle-coordinator'

import { Autopilot } from './workers/autopilot'
import { CompactionWorker } from './workers/compaction-worker'
import { ApprovalWorker } from './workers/approval-worker'
import type { AgentVariant } from './agents'
import { ChatTitleWorker } from './workers/chat-title-worker'
import { UserPresenceWorker } from './workers/user-presence-worker'
import { ArtifactSyncWorker } from './workers/artifact-sync-worker'

// Execution
import { ExecutionManager, ExecutionManagerLive } from './execution/execution-manager'
import { BrowserServiceLive } from './services/browser-service'
import { registerApprovalBridge } from './execution/approval-bridge'

// Persistence
import { ChatPersistence } from './persistence/chat-persistence-service'

// Utils
import { collectSessionContext } from './util/collect-session-context'

// Providers
import { initializeProviderState, loadConfig, onTrace } from '@magnitudedev/providers'
import { writeTrace, initTraceSession } from '@magnitudedev/tracing'
import { join } from 'path'
import { createMemoryExtractionJob, drainPendingJobsOnStartup, spawnDetachedMemoryExtractionWorker, writePendingJobSync } from './memory/job-queue'
import { MEMORY_RELATIVE_PATH } from './memory/memory-file'

// =============================================================================
// Agent
// =============================================================================

export const CodingAgent = Agent.define<AppEvent>()({
  name: 'CodingAgent',

  projections: [
    SessionContextProjection,
    ForkProjection,
    CompactionProjection,
    WorkingStateProjection,
    TurnProjection,
    CanonicalTurnProjection,

    ArtifactProjection,
    AgentRegistryProjection,

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
    ForkOrchestrator,
    LifecycleCoordinator,
    Autopilot,
    CompactionWorker,
    ApprovalWorker,

    ArtifactSyncWorker,

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
      forks: ForkProjection,

      artifacts: ArtifactProjection,
      agentRegistry: AgentRegistryProjection,
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
   * Enable LLM call tracing to ~/.magnitude/traces/
   */
  debug?: boolean

  /**
   * Provide a pre-built session context instead of collecting from the local environment.
   * Useful for evals / headless runs where the agent operates in a container.
   */
  sessionContext?: SessionContext
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
  // Initialize provider state from stored config / env vars
  await initializeProviderState()

  if (!hasDrainedPendingMemoryJobs) {
    hasDrainedPendingMemoryJobs = true
    const cfg = loadConfig()
    if (cfg.memory !== false) {
      drainPendingJobsOnStartup().catch(() => {})
    }
  }

  // Enable tracing in debug mode
  if (options.debug) {
    const traceSessionId = new Date().toISOString().replace(/:/g, '-').replace(/\.\d{3}Z$/, 'Z')
    initTraceSession(traceSessionId, { cwd: process.cwd(), platform: process.platform, gitBranch: null })
    onTrace(writeTrace)
  }

  const layer = Layer.mergeAll(ExecutionManagerLive, BrowserServiceLive, options.persistence)
  const client = await CodingAgent.createClient(layer)

  await client.runEffect(Effect.gen(function* () {
    const persistence = yield* ChatPersistence
    const hydrationContext = yield* HydrationContext
    const eventSink = yield* EventSinkTag<AppEvent>()

    // Bridge approval state into display and working-state projections
    yield* registerApprovalBridge

    const events = yield* persistence.loadEvents()

    if (events.length === 0) {
      // New session
      const context = options.sessionContext ?? (yield* Effect.promise(() => collectSessionContext()))

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
      const forkProjection = yield* ForkProjection.Tag

      // Create root sandbox (hydration happens lazily in execute())
      yield* executionManager.initFork(null, 'orchestrator')

      // Create sandboxes for all active (running) forks
      const forkState = yield* forkProjection.get
      for (const [forkId, fork] of forkState.forks) {
        if (fork.status !== 'running') continue
        yield* executionManager.initFork(forkId, (fork.role ?? 'orchestrator') as AgentVariant)
      }

      // Complete any orphaned forks (running forks with no fork_completed persisted —
      // happens when the process is killed mid-run). Publishing fork_completed triggers
      // ForkOrchestrator to dispose the sandbox and schedule fork_removed, and interrupts
      // any idle Cortex fibers. These events are persisted so subsequent loads are clean.
      for (const [forkId, fork] of forkState.forks) {
        if (fork.status !== 'running') continue
        yield* Effect.promise(() => client.send({
          type: 'fork_completed',
          forkId,
          parentForkId: fork.parentForkId,
          result: { interrupted: true },
        }))
      }
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
    const cfg = loadConfig()
    const memoryEnabled = cfg.memory !== false

    let jobPath: string | null = null
    if (memoryEnabled) {
      try {
        const metadata = await client.runEffect(Effect.gen(function* () {
          const persistence = yield* ChatPersistence
          return yield* persistence.getSessionMetadata()
        }))
        const cwd = metadata.workingDirectory
        const sessionId = metadata.sessionId
        const eventsPath = join(process.env.HOME || '', '.magnitude', 'sessions', sessionId, 'events.jsonl')
        const memoryPath = join(cwd, MEMORY_RELATIVE_PATH)
        const job = createMemoryExtractionJob({ sessionId, cwd, eventsPath, memoryPath })
        jobPath = writePendingJobSync(job)
      } catch {}
    }

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


