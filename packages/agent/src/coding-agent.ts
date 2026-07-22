/**
 * Coding Agent
 *
 * A minimal coding agent that:
 * - Uses event-core architecture (projections, workers, signals)
 * - Uses native tool calling via TurnEngine
 * - Has a shell command tool for executing commands
 * - Supports session persistence and hydration
 */

import { Context, Data, Effect, Fiber, Layer, Option, Stream } from 'effect'
import { EventEngine, Introspection, Surface } from '@magnitudedev/event-core'
import { AmbientServiceTag, HydrationContext, WorkerBusTag, type AmbientService } from '@magnitudedev/event-core'
import type { FrameworkError } from '@magnitudedev/event-core'
import type { AppEvent, SessionContext } from './events'
import type { AgentIntrospection, AgentIntrospectionError } from './introspection/session'

// Projections
import { SessionContextProjection } from './projections/session-context'
import { TurnProjection } from './projections/turn'
import { HarnessStateProjection } from './projections/harness-state'
import { AgentToolkitProjection } from './projections/agent-toolkit'
import { DetachedProcessProjection } from './projections/detached-process'
import { WindowProjection } from './window'
import { WorkerActivityProjection } from './projections/worker-activity'
import { DisplayTimelineProjection } from './display'
import { AutopilotStateProjection } from './projections/autopilot-state'

import { AgentRoutingProjection } from './projections/agent-routing'
import { AgentLifecycleProjection } from './projections/agent-lifecycle'
import { TaskGraphProjection } from './projections/task-graph'
import { TaskAssignmentProjection } from './projections/task-assignment'
import { CompactionProjection } from './projections/compaction'

import { ConversationProjection } from './projections/conversation'
import { OutboundMessagesProjection } from './projections/outbound-messages'
import { UserMessageResolutionProjection } from './projections/user-message-resolution'
import { ChatTitleProjection } from './projections/chat-title'
import { AtifProjection } from './projections/atif/projection'
import { GoalProjection } from './projections/goal'


// Workers
import { TurnController } from './workers/turn-controller'
import { Cortex } from './workers/cortex'
import { AgentLifecycle } from './workers/agent-lifecycle'
import { RetryController } from './workers/retry-controller'
import { LifecycleCoordinator } from './workers/lifecycle-coordinator'

// Runtime
import { EffectLoggerLayer } from './runtime/effect-logger'
import { loadProjectionSnapshotHydrationEvents } from './runtime/projection-snapshot-hydration'

// TEMPORARILY DISABLED: Autopilot worker registration.
// import { Autopilot } from './workers/autopilot'
import { ObserverStateLive } from './observer/state'
// TEMPORARILY DISABLED: Observer worker registration.
// import { ObserverWorker } from './observer'
import { CompactionWorker } from './compaction/worker'
import { isRoleId, type RoleId } from './agents/role-validation'
import { getForkInfo } from './agents/registry'
import { ROLE_TO_SLOT } from '@magnitudedev/roles'
import { FileMentionResolver } from './workers/file-mention-resolver'
import { ChatTitleServiceLive } from './workers/chat-title-service'
import { ChatTitleWorker } from './workers/chat-title-worker'
import { AtifWriter } from './workers/atif-writer'
import { ProcessMetricsWorker } from './workers/process-metrics'
import { FsLive } from './services/fs'

// Execution
import { ExecutionManager } from './execution/types'
import { ExecutionManagerLive } from './execution/execution-manager'

import { FetchHttpClient } from '@effect/platform'
import type { PlatformError } from '@effect/platform/Error'

// Persistence
import { ChatPersistence, type PersistenceError } from './persistence/chat-persistence-service'
import { makeChatAddressedEntryStoreLayer } from './persistence/addressed-entry-store'
import {
  DisplayViewRuntime,
  DisplayViewRuntimeLive,
  type DisplayViewNotFoundError,
  type DisplayViewRuntimeError,
} from './display-view'

// Utils
import { collectSessionContext } from './util/collect-session-context'
// import { isVcsAvailable } from './util/vcs-availability'  // VCS currently disabled — see session options below

// Engine layers

import { AgentModelResolverLive } from './model/model-resolver'

// Config & Auth
import { ProviderClient, SlotIdSchema, type ProviderClientShape } from '@magnitudedev/sdk'
import type { DisplayViewShape, DisplayViewSnapshot } from '@magnitudedev/protocol'
import type { ForkTurnState } from './projections/turn'
import type { AgentLifecycleState } from './projections/agent-lifecycle'

import { MagnitudeStorage, type MagnitudeStorageShape } from '@magnitudedev/storage'
import { initLogger, logger } from '@magnitudedev/logger'
import { BunFileSystem, BunPath } from '@effect/platform-bun'
import { initTraceSession } from '@magnitudedev/tracing'
import { MAGNITUDE_VERSION } from '@magnitudedev/version'

import { publishSessionOptions, SessionOptionsAmbient } from './ambient/session-ambient'
import { ConfigAmbient, getSlotConfig, type ConfigState } from './ambient/config-ambient'
import { loadSkills, skillLoadDiagnosticLogFields, type Skill, type SkillLoadDiagnostic } from '@magnitudedev/skills'
import { publishSkills } from './ambient/skills-ambient'
import { publishAtifConfig, DEFAULT_ATIF_CONFIG } from './ambient/atif-ambient'
import { publishInitialTask as publishInitialTaskAmbient } from './ambient/initial-task-ambient'
import { ToolUniverseSourceLive } from './tools/tool-universe-live'

// =============================================================================
// Coding Agent
// =============================================================================

const logSkillLoadDiagnostic = (diagnostic: SkillLoadDiagnostic): void => {
  logger.error(skillLoadDiagnosticLogFields(diagnostic), 'Failed to load skill')
}

const loadRuntimeSkills = (cwd: string) =>
  loadSkills(cwd, { onDiagnostic: logSkillLoadDiagnostic })

export const CodingAgent = EventEngine.make<AppEvent>()({
  name: 'CodingAgent',
  schemaVersion: MAGNITUDE_VERSION,

  projections: [
    SessionContextProjection,
    AgentRoutingProjection,
    AgentLifecycleProjection,
    AgentToolkitProjection,
    GoalProjection,
    TaskGraphProjection,
    CompactionProjection,
    TurnProjection,
    HarnessStateProjection,
    DetachedProcessProjection,

    WorkerActivityProjection,
    OutboundMessagesProjection,
    UserMessageResolutionProjection,

    WindowProjection,
    TaskAssignmentProjection,
    DisplayTimelineProjection,
    ConversationProjection,
    AutopilotStateProjection,
    AtifProjection,
    ChatTitleProjection,
  ],

  workers: [
    TurnController,
    Cortex,
    AgentLifecycle,
    RetryController,
    LifecycleCoordinator,
    // TEMPORARILY DISABLED: Autopilot.
    // Implementation remains in workers/autopilot.ts for future re-enable.
    // Autopilot,
    // TEMPORARILY DISABLED: Observer.
    // Implementation remains in observer/ for future re-enable.
    // ObserverWorker,
    CompactionWorker,

    FileMentionResolver,
    ChatTitleWorker,
    AtifWriter,
    ProcessMetricsWorker,
  ],

})

type CodingAgentEngine = EventEngine.Shape<AppEvent, typeof CodingAgent.expose, typeof CodingAgent.projections>

const CodingAgentEngineTag = Context.GenericTag<CodingAgentEngine>('EventEngine')

// =============================================================================
// Coding Agent Service
// =============================================================================

export interface CodingAgentService {
  readonly events: Stream.Stream<AppEvent>
  readonly errors: Stream.Stream<FrameworkError>
  readonly initialize: () => Effect.Effect<void>
  readonly send: (event: AppEvent) => Effect.Effect<void>
  readonly interrupt: () => Effect.Effect<void>
  readonly publishInitialTask: (task: string | null) => Effect.Effect<void>
  readonly introspectionChanges: (forkId: string | null) => Stream.Stream<AgentIntrospection, AgentIntrospectionError>
}

export const CodingAgentTag = Context.GenericTag<CodingAgentService>('CodingAgent')

// =============================================================================
// Client Factory
// =============================================================================

export interface CreateClientOptions {
  /**
   * Persistence service for session storage and hydration.
   */
  persistence: Layer.Layer<ChatPersistence, never, never>

  /**
   * Storage shape for config, sessions, memory, and logs.
   */
  storage: MagnitudeStorageShape

  /**
   * Enable LLM call tracing to ~/.magnitude/traces/
   */
  debug?: boolean

  /**
   * Provide a pre-built session context instead of collecting from the local environment.
   * Useful for evals / headless runs where the agent operates in a container.
   */
  sessionContext?: Omit<SessionContext, 'scratchpadPath'>

  /**
   * Pre-built provider client from the SDK boundary. The agent wraps its
   * catalog with a file-backed cache and provides it as `ProviderClient`.
   */
  providerClient: ProviderClientShape

  /** Replaying coherent snapshots from ACN's authoritative model state. */
  modelConfigurations: Stream.Stream<ConfigState>
  /** ACN-owned authoritative persistence/publication for a runtime-invalidated bound effort. */
  applyReasoningEffortFallback?: (
    input: import('./model/model-resolver').ReasoningEffortFallbackInput,
  ) => Effect.Effect<void, unknown>

  /**
   * Disable shell command classification safeguards for this runtime only.
   */
  disableShellSafeguards?: boolean

  /**
   * Disable working-directory boundary safeguards for this runtime only.
   */
  disableCwdSafeguards?: boolean

  /**
   * Session ID to use for trace recording. When provided with debug mode,
   * the trace folder uses this ID instead of a date-based string.
   */
  sessionId?: string

  /**
   * Write ATIF trajectory to the specified path on session end.
   */
  atifPath?: string

  /**
   * Run in headless mode — no TUI, no user present.
   * Adds a section to the system prompt telling the agent to operate autonomously.
   */
  headless?: boolean

  /**
   * Solo mode — removes task/worker tools from the leader toolkit.
   * The leader cannot create tasks, spawn workers, or message workers.
   */
  solo?: boolean

  /**
   * Override the leader system prompt with the given text.
   * When set, used in place of the role's shipped prompt template.
   */
  systemPromptOverride?: string
}

export class CodingAgentStartupError extends Data.TaggedError('CodingAgentStartupError')<{
  readonly reason: string
}> {}

function makeCodingAgentLive(options: CreateClientOptions) {
  return Layer.scoped(
    CodingAgentTag,
    Effect.gen(function* () {
      const engine = yield* CodingAgentEngineTag
      const persistence = yield* ChatPersistence
      const workerBus = yield* WorkerBusTag<AppEvent>()
      const execManager = yield* ExecutionManager
      const ambientService = yield* AmbientServiceTag
      const hydrationContext = yield* HydrationContext
      const sessionContextProjection = yield* SessionContextProjection.Tag
      const agentRoutingProjection = yield* AgentRoutingProjection.Tag
      const agentLifecycleProjection = yield* AgentLifecycleProjection.Tag
      const taskGraphProjection = yield* TaskGraphProjection.Tag
      const goalProjection = yield* GoalProjection.Tag
      const turnProjection = yield* TurnProjection.Tag
      const windowProjection = yield* WindowProjection.Tag
      const compactionProjection = yield* CompactionProjection.Tag
      const conversationProjection = yield* ConversationProjection.Tag
      const displayTimelineProjection = yield* DisplayTimelineProjection.Tag
      const harnessStateProjection = yield* HarnessStateProjection.Tag
      const magnitudeClient = yield* ProviderClient
      const addressedIntrospectionRegistry = yield* Effect.serviceOption(Introspection.AddressedIntrospectionRegistry)
      const runtimeIntrospector = yield* Effect.serviceOption(Introspection.RuntimeIntrospector)
      const provideAmbient = <A, E>(effect: Effect.Effect<A, E, AmbientService>): Effect.Effect<A, E> =>
        effect.pipe(Effect.provideService(AmbientServiceTag, ambientService))

      const initializeFork = (forkId: string | null, role: RoleId): Effect.Effect<void> =>
        execManager.initFork(forkId, role).pipe(
          Effect.provideService(ChatPersistence, persistence),
          Effect.provideService(WorkerBusTag<AppEvent>(), workerBus),
          Effect.provideService(SessionContextProjection.Tag, sessionContextProjection),
          Effect.provideService(AgentRoutingProjection.Tag, agentRoutingProjection),
          Effect.provideService(AgentLifecycleProjection.Tag, agentLifecycleProjection),
          Effect.provideService(TaskGraphProjection.Tag, taskGraphProjection),
          Effect.provideService(GoalProjection.Tag, goalProjection),
          Effect.provideService(TurnProjection.Tag, turnProjection),
          Effect.provideService(WindowProjection.Tag, windowProjection),
          Effect.provideService(ConversationProjection.Tag, conversationProjection)
        )

      const provideIntrospectionRequirements = <A, E, R>(effect: Effect.Effect<A, E, R>) => {
        const withRuntime = Option.isSome(runtimeIntrospector)
          ? effect.pipe(
              Effect.provideService(
                Introspection.RuntimeIntrospector,
                runtimeIntrospector.value
              )
            )
          : effect
        const withAddressed = Option.isSome(addressedIntrospectionRegistry)
            ? withRuntime.pipe(
                Effect.provideService(
                  Introspection.AddressedIntrospectionRegistry,
                  addressedIntrospectionRegistry.value
                )
              )
            : withRuntime
        return withAddressed.pipe(
          Effect.provideService(AmbientServiceTag, ambientService),
          Effect.provideService(DisplayTimelineProjection.Tag, displayTimelineProjection),
          Effect.provideService(AgentLifecycleProjection.Tag, agentLifecycleProjection),
          Effect.provideService(WindowProjection.Tag, windowProjection),
          Effect.provideService(CompactionProjection.Tag, compactionProjection)
        )
      }

      const applyConfig = (state: ConfigState): Effect.Effect<void> =>
        ambientService.update(ConfigAmbient, state)
      const loadInitialConfig: Effect.Effect<void> = options.modelConfigurations.pipe(
        Stream.runHead,
        Effect.flatMap(Option.match({ onNone: () => Effect.void, onSome: applyConfig })),
      )

      yield* Effect.acquireRelease(
        Effect.forkScoped(Stream.runForEach(options.modelConfigurations, applyConfig)),
        Fiber.interrupt,
      )

      // Root resources are app-owned, not event-core-owned. LifecycleCoordinator
      // persists anything emitted by this cleanup when the engine scope closes.
      yield* Effect.acquireRelease(
        Effect.void,
        () => execManager.disposeFork(null).pipe(
          Effect.catchAllCause((cause) =>
            Effect.sync(() => logger.warn({ cause }, 'Failed to dispose root fork'))
          )
        )
      )

      const initialize = Effect.fn("agent.initialize")(function* (): Effect.fn.Return<
        void,
        PersistenceError | PlatformError
      > {
        const sessionMetadata = yield* persistence.getSessionMetadata().pipe(
          Effect.catchTag('PersistenceError', (e) => {
            logger.warn(`Failed to load session metadata: ${e.message}`)
            return Effect.succeed(null)
          })
        )

        if (sessionMetadata) {
          initLogger(sessionMetadata.sessionId)
        }

        yield* provideAmbient(publishSessionOptions({
          sessionId: sessionMetadata?.sessionId ?? options.sessionId ?? '',
          disableShellSafeguards: options.disableShellSafeguards ?? false,
          disableCwdSafeguards: options.disableCwdSafeguards ?? false,
          timezone: options.sessionContext?.timezone ?? null,
          vcsAvailable: false,
          headless: options.headless ?? false,
          solo: options.solo ?? false,
          systemPromptOverride: options.systemPromptOverride,
        }))

        const hydration = yield* loadProjectionSnapshotHydrationEvents({
          persistence,
          engine,
          beforeSnapshotCommit: hydrationContext.setHydrating(true),
          warn: ({ fields, message }) => logger.warn(fields, message)
        })
        const restoredFromSnapshot = hydration.restoredFromSnapshot
        const events = hydration.events

        if (!restoredFromSnapshot && events.length === 0) {
          const baseContext = options.sessionContext ?? (yield* Effect.tryPromise(() =>
            collectSessionContext({ cwd: process.cwd(), storage: options.storage })
          ).pipe(
            Effect.catchTag('UnknownException', (e) => {
              logger.error({ err: e }, 'Failed to collect session context')
              return Effect.succeed({
                cwd: process.cwd(),
                platform: process.platform === 'darwin' ? 'macos' as const : process.platform === 'win32' ? 'windows' as const : 'linux' as const,
                shell: process.env.SHELL?.split('/').pop() || 'bash',
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                username: process.env.USER || 'unknown',
                fullName: null,
                git: null,
                folderStructure: '(failed to collect folder structure)',
                agentsFile: null,
                skills: null,
              })
            })
          ))

          const metadata = yield* persistence.getSessionMetadata()
          const scratchpadPath = yield* options.storage.sessions.createSessionScratchpad(metadata.sessionId)
          const context: SessionContext = {
            ...baseContext,
            scratchpadPath,
          }

          yield* engine.send({
            type: 'session_initialized',
            forkId: null,
            context
          })

          if (options.atifPath) {
            yield* provideAmbient(publishAtifConfig({
              enabled: true,
              writeFile: true,
              filePath: options.atifPath,
              streamSteps: false,
              stepsPath: null,
            }))
          } else {
            yield* provideAmbient(publishAtifConfig(DEFAULT_ATIF_CONFIG))
          }

          yield* loadInitialConfig

          const skills = yield* Effect.tryPromise(() => loadRuntimeSkills(process.cwd())).pipe(
            Effect.catchTag('UnknownException', (error) =>
              Effect.sync(() => logger.error({
                error: error.message,
                cause: error.cause,
              }, 'Failed to load skills during session initialization')).pipe(
                Effect.as(new Map<string, Skill>()),
              )
            ),
          )
          yield* provideAmbient(publishSkills(skills))

          return
        }

        const metadata = yield* persistence.getSessionMetadata()
        yield* options.storage.sessions.createSessionScratchpad(metadata.sessionId)

        yield* hydrationContext.setHydrating(true)

        if (options.atifPath) {
          yield* provideAmbient(publishAtifConfig({
            enabled: true,
            writeFile: true,
            filePath: options.atifPath,
            streamSteps: false,
            stepsPath: null,
          }))
        } else {
          yield* provideAmbient(publishAtifConfig(DEFAULT_ATIF_CONFIG))
        }

        yield* loadInitialConfig
        const skills = yield* Effect.tryPromise(() => loadRuntimeSkills(process.cwd())).pipe(
          Effect.catchTag('UnknownException', (error) =>
            Effect.sync(() => logger.error({
              error: error.message,
              cause: error.cause,
            }, 'Failed to load skills during session hydration')).pipe(
              Effect.as(new Map<string, Skill>()),
            )
          ),
        )
        yield* provideAmbient(publishSkills(skills))

        for (const event of events) {
          yield* engine.send(event)
        }

        yield* Effect.sleep('50 millis')
        yield* hydrationContext.setHydrating(false)

        yield* initializeFork(null, 'leader')

        const agentState = yield* agentLifecycleProjection.get
        for (const [, agent] of agentState.agents) {
          if (!isRoleId(agent.role)) {
            continue
          }
          yield* initializeFork(agent.forkId, agent.role)
        }

        for (const [, agent] of agentState.agents) {
          const forkTurnState = yield* turnProjection.getFork(agent.forkId)
          if (forkTurnState._tag === 'active' || forkTurnState._tag === 'interrupting') {
            yield* engine.send({
              type: 'turn_outcome',
              forkId: agent.forkId,
              turnId: forkTurnState.turnId,
              chainId: forkTurnState.chainId,
              strategyId: 'native',
              outcome: { _tag: 'Cancelled', reason: { _tag: 'WorkerKilled' }, requestId: null },
              inputTokens: null,
              outputTokens: null,
              cacheReadTokens: null,
              cacheWriteTokens: null,
              cost: null,
              providerId: null,
              modelId: null,
            })
          }
        }

        const rootTurnState = yield* turnProjection.getFork(null)
        if (rootTurnState._tag === 'active' || rootTurnState._tag === 'interrupting') {
          yield* engine.send({
            type: 'turn_outcome',
            forkId: null,
            turnId: rootTurnState.turnId,
            chainId: rootTurnState.chainId,
            strategyId: 'native',
            outcome: { _tag: 'Cancelled', reason: { _tag: 'WorkerKilled' }, requestId: null },
            inputTokens: null,
            outputTokens: null,
            cacheReadTokens: null,
            cacheWriteTokens: null,
            cost: null,
            providerId: null,
            modelId: null,
          })
        }

      }, Effect.orDie)

      const introspectionChanges = (forkId: string | null) =>
        Stream.unwrap(
          Effect.gen(function* () {
            const {
              getAgentIntrospection,
              createAgentIntrospectionChanges,
            } = yield* Effect.promise(() => import('./introspection/session'))
            const initial = yield* provideIntrospectionRequirements(getAgentIntrospection(forkId))
            const stream = yield* provideIntrospectionRequirements(createAgentIntrospectionChanges(forkId))
            return Stream.concat(
              Stream.succeed(initial),
              stream
            )
          })
        )

      // Self-heal: monitor turn_outcome events for ProviderNotReady/OutOfSync.
      // When a slot with a user override fails, clear the override and refresh
      // config so the next turn uses the default model. Per spec §9.6.
      const selfHeal = Stream.runForEach(engine.events, (event) =>
        Effect.gen(function* () {
          if (event.type !== 'turn_outcome') return
          const outcome = event.outcome
          if (outcome._tag !== 'ProviderNotReady') return
          if (outcome.detail._tag !== 'OutOfSync') return

          const agentState = yield* agentLifecycleProjection.get
          const forkInfo = getForkInfo(agentState, event.forkId)
          if (!forkInfo || !forkInfo.roleId) return

          const slotId = ROLE_TO_SLOT[forkInfo.roleId]
          const configState = ambientService.getValue(ConfigAmbient)
          const slotConfig = getSlotConfig(configState, slotId)
          if (!slotConfig || !slotConfig.isUserOverride) return
          if (slotConfig.providerId === 'local') return

          yield* options.storage.config.updateModelSlot(SlotIdSchema.make(slotId), Option.none()).pipe(
            Effect.catchAll(() =>
              Effect.logWarning(`Self-heal: failed to clear override for slot ${slotId}`)
            ),
          )
        })
      ).pipe(Effect.ignoreLogged)

      yield* Effect.acquireRelease(
        Effect.forkScoped(selfHeal),
        (fiber) => Fiber.interrupt(fiber),
      )

      return {
        events: engine.events,
        errors: engine.errors,
        initialize,
        send: (event) => engine.send(event),
        interrupt: () => engine.interrupt(),
        publishInitialTask: (task) => provideAmbient(publishInitialTaskAmbient(task)),
        introspectionChanges,
      } satisfies CodingAgentService
    })
  )
}

/**
 * Create a CodingAgent session handle with persistence.
 *
 * Loads events from persistence on startup:
 * - If events exist: hydrates projections from persisted state
 * - If no events: initializes a new session
 */
export interface CodingAgentSession {
  readonly on: {
    readonly restoreQueuedMessages: Stream.Stream<{
      readonly forkId: string | null
      readonly messages: ReadonlyArray<{
        readonly id: string
        readonly content: string
        readonly taskMode: boolean
      }>
    }>
  }
  readonly state: {
    readonly turn: {
      readonly getFork: (forkId: string | null) => Effect.Effect<ForkTurnState>
      readonly subscribeFork: (forkId: string | null) => Stream.Stream<ForkTurnState>
    }
    readonly agentStatus: {
      readonly get: () => Effect.Effect<AgentLifecycleState>
      readonly subscribe: Stream.Stream<AgentLifecycleState>
    }
  }
  readonly displayView: {
    readonly stream: (viewId: string) => Stream.Stream<DisplayViewSnapshot, DisplayViewNotFoundError | DisplayViewRuntimeError>
    readonly snapshot: (viewId: string) => Effect.Effect<DisplayViewSnapshot, DisplayViewNotFoundError | DisplayViewRuntimeError>
    readonly setShape: (viewId: string, shape: DisplayViewShape) => Effect.Effect<void, DisplayViewRuntimeError>
    readonly close: (viewId: string) => Effect.Effect<void>
  }
  readonly send: (event: AppEvent) => Effect.Effect<void>
  readonly interrupt: () => Effect.Effect<void>
  readonly publishInitialTask: (task: string | null) => Effect.Effect<void>
  readonly onEvent: Stream.Stream<AppEvent>
  readonly onError: Stream.Stream<FrameworkError>
  readonly subscribeIntrospection: (forkId: string | null) => Stream.Stream<AgentIntrospection, AgentIntrospectionError>
}

export type CodingAgentClient = CodingAgentSession

export function createCodingAgentSession(options: CreateClientOptions) {
  return Effect.gen(function* () {
  // ACN owns the authoritative catalog; sessions consume it without adding a
  // second cache or refresh policy.
  const providerClientLayer = Layer.succeed(ProviderClient, options.providerClient)
  const agentModelResolverLayer = Layer.provide(
    AgentModelResolverLive(options.debug, options.applyReasoningEffortFallback),
    providerClientLayer,
  )
  const chatTitleServiceLayer = Layer.provide(
    ChatTitleServiceLive,
    Layer.mergeAll(agentModelResolverLayer, FetchHttpClient.layer, options.persistence),
  )
  const introspectionLayer = options.debug
    ? Introspection.AddressedIntrospectionRegistryLive
    : Layer.empty
  const persistenceServicesLayer = Layer.provideMerge(
    makeChatAddressedEntryStoreLayer(options.storage, options.sessionId),
    options.persistence
  )

  // Enable tracing in debug mode
  const traceSessionId = options.sessionId ?? new Date().toISOString().replace(/:/g, '-').replace(/\.\d{3}Z$/, 'Z')
  if (options.debug) {
    initTraceSession(traceSessionId, { cwd: process.cwd(), platform: process.platform, gitBranch: null })
  }

  const baseLayer = Layer.mergeAll(
    EffectLoggerLayer,
    ExecutionManagerLive,

    agentModelResolverLayer,
    providerClientLayer,
    chatTitleServiceLayer,
    ObserverStateLive,

    FetchHttpClient.layer,
    FsLive,
    introspectionLayer,
    persistenceServicesLayer,
    Layer.succeed(MagnitudeStorage, options.storage),
    BunFileSystem.layer,
    BunPath.layer,
    ToolUniverseSourceLive,
  )
  // All worker requirements are supplied by baseLayer. The EventEngine worker
  // tuple currently widens that requirement parameter to `any`, so constrain
  // it at this composition boundary instead of leaking `any` into Surface.host.
  const engineLayer = Layer.provideMerge(
    CodingAgent.EngineLayer as Layer.Layer<any, any, never>,
    baseLayer,
  )
  const withDisplayRuntime = Layer.provideMerge(DisplayViewRuntimeLive, engineLayer)
  const appLayer = Layer.provideMerge(
    makeCodingAgentLive(options),
    withDisplayRuntime
  )
  const runtimeLayer = Layer.provideMerge(
    Layer.scopedDiscard(
      Effect.flatMap(CodingAgentTag, (agent) => agent.initialize())
    ),
    appLayer
  )

  return yield* Surface.effectClient(Surface.host({
    layer: runtimeLayer,
    on: {
      restoreQueuedMessages: Surface.signal(DisplayTimelineProjection.signals.restoreQueuedMessages),
    },
    state: {
      turn: Surface.state(TurnProjection),
      agentStatus: Surface.state(AgentLifecycleProjection),
    },
    displayView: {
      stream: Surface.signal((viewId: string) =>
        Stream.unwrap(
          Effect.gen(function* () {
            const runtime = yield* DisplayViewRuntime
            return runtime.stream(viewId)
          })
        )
      ),
      snapshot: Surface.command((viewId: string) =>
        Effect.gen(function* () {
          const runtime = yield* DisplayViewRuntime
          return yield* runtime.snapshot(viewId)
        })
      ),
      setShape: Surface.command((viewId: string, shape: DisplayViewShape) =>
        Effect.flatMap(DisplayViewRuntime, (runtime) =>
          runtime.setShape(viewId, shape)
        )
      ),
      close: Surface.command((viewId: string) =>
        Effect.flatMap(DisplayViewRuntime, (runtime) => runtime.close(viewId))
      ),
    },
    send: Surface.command((event: AppEvent) =>
      Effect.flatMap(CodingAgentTag, (agent) => agent.send(event))
    ),
    interrupt: Surface.command(() =>
      Effect.flatMap(CodingAgentTag, (agent) => agent.interrupt())
    ),
    publishInitialTask: Surface.command((task: string | null) =>
      Effect.flatMap(CodingAgentTag, (agent) => agent.publishInitialTask(task))
    ),
    onEvent: Surface.signal(
      Stream.unwrap(
        Effect.map(CodingAgentTag, (agent) => agent.events)
      )
    ),
    onError: Surface.signal(
      Stream.unwrap(
        Effect.map(CodingAgentTag, (agent) => agent.errors)
      )
    ),
    subscribeIntrospection: Surface.signal((forkId: string | null) =>
      Stream.unwrap(
        Effect.map(CodingAgentTag, (agent) => agent.introspectionChanges(forkId))
      )
    ),
  }))
  })
}

export const createCodingAgentClient = createCodingAgentSession
