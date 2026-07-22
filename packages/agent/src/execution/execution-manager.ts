/**
 * ExecutionManager
 *
 * Owns per-fork lifecycle: init, dispose, fork, layer caching, observables.
 */

import * as path from 'path'
import { Effect, Layer } from 'effect'
import { ToolInterceptorTag, type ToolInterceptor } from './permission-gate'
import { Fork, Projection, WorkerBusTag, AmbientServiceTag, type WorkerBusService } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { isToolKey, type ToolKey } from '../tools/toolkits'

import { isRoleId, type RoleId } from '../agents/role-validation'
import { getAgentDefinition } from '../agents/registry'
import { buildPolicyInterceptor, type AgentResolver } from './permission-gate'
export { IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from './types'

import { AgentStateReaderTag, type AgentStateReader } from '../tools/fork'
import { AgentRegistryStateReaderTag, type AgentRegistryStateReader } from '../tools/agent-registry-reader'
import { WindowStateReaderTag, type WindowStateReader } from '../tools/window-reader'
import { buildCloneContext, buildSpawnContext } from '../prompts/fork-context'
import type { JsonSchema } from '../prompts/fork-context'
import { ConversationStateReaderTag, type ConversationStateReader } from '../tools/memory-reader'
import { TaskGraphStateReaderTag, canCompleteRecord, getChildRecords, canAssignRecord, collectSubtreeRecords } from '../tools/task-reader'
import { GoalStateReaderTag, type GoalStateReader } from '../tools/goal'
import { ConversationProjection, type ConversationState } from '../projections/conversation'
import { createId } from '../util/id'
import { logger } from '@magnitudedev/logger'

import { AgentRoutingProjection, type AgentRoutingState } from '../projections/agent-routing'
import { AgentLifecycleProjection, type AgentLifecycleState, getAgentByForkId } from '../projections/agent-lifecycle'
import { TurnProjection, type ForkTurnState } from '../projections/turn'
import { SessionContextProjection, type SessionContextState } from '../projections/session-context'
import { TaskGraphProjection, type TaskGraphState } from '../projections/task-graph'
import { WindowProjection, type ForkWindowState } from '../window'
import { GoalProjection, type GoalState } from '../projections/goal'

import type { RoleDefinition } from '@magnitudedev/roles'
import type { BoundObservable } from '../observables/types'
import { bindObservable } from '../observables/types'
import { ProjectionReaderTag, type ProjectionReader } from '../observables/projection-reader'
import { PolicyContextProviderTag } from '../agents/types'
import { SessionOptionsAmbient, type SessionOptions } from '../ambient/session-ambient'
import { createPolicyContextProvider } from '../agents/policy-context'
import { ExecutionManager } from './types'
import type { ExecutionManagerService } from './types'
import type { ForkLayer } from './fork-layer'
import { WorkingDirectoryTag } from './working-directory'
import { DetachedShellRegistry, type DetachedShellRegistryService } from '../process/detached-process-registry'
import { makeDetachedShellRegistryService } from '../process/detached-process-registry-live'

import { ChatPersistence } from '../persistence/chat-persistence-service'
import { ShadowVcs, makeShadowVcsLayer, VcsFsLive, makeNoOpVcsLayer } from '@magnitudedev/vcs'

const { ForkContext } = Fork

type AgentDef = RoleDefinition
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


// =============================================================================
// Implementation
// =============================================================================

/**
 * Build the unified Effect layer for a fork — covers tool execution, interceptor, and emit.
 * Tools use reader services, interceptor uses PolicyContextProvider.
 */
function makeForkLayers(
  forkId: string | null,
  roleId: string,

  sessionContextProjection: SessionContextProjectionInstance,
  agentProjection: AgentRoutingProjectionInstance,
  agentLifecycleProjection: AgentLifecycleProjectionInstance,
  workingStateProjection: TurnProjectionInstance,
  taskGraphProjection: TaskGraphProjectionInstance,
  goalProjection: GoalProjectionInstance,
  windowProjection: WindowProjectionInstance,

  conversationProjection: ConversationProjectionInstance,
  persistenceLayer: Layer.Layer<ChatPersistence, never, never>,
  policyInterceptor: ReturnType<typeof buildPolicyInterceptor>,

  cwd: string,
  scratchpadPath: string,
  sessionOptions: SessionOptions,
  detachedShellRegistryService: DetachedShellRegistryService,
  vcsLayer: Layer.Layer<ShadowVcs>,
) {
  const agentRegistryStateReaderLayer = Layer.succeed(AgentRegistryStateReaderTag, {
    getState: () => agentLifecycleProjection.get
  } satisfies AgentRegistryStateReader)

  const conversationStateReaderLayer = Layer.succeed(ConversationStateReaderTag, {
    getState: () => conversationProjection.get
  } satisfies ConversationStateReader)

  const agentStateReaderLayer = Layer.succeed(AgentStateReaderTag, {
    getAgentState: () => agentLifecycleProjection.get,
    getAgent: (agentId: string) => Effect.map(agentLifecycleProjection.get, (state) => state.agents.get(agentId)),
  } satisfies AgentStateReader)

  const windowStateReaderLayer = Layer.succeed(WindowStateReaderTag, {
    getWindowState: (targetForkId) => windowProjection.getFork(targetForkId),
  } satisfies WindowStateReader)

  const taskGraphReaderLayer = Layer.succeed(TaskGraphStateReaderTag, {
    getTask: (id) => Effect.map(taskGraphProjection.get, (s) => s.tasks.get(id)),
    getState: () => taskGraphProjection.get,
    getChildren: (id) => Effect.map(taskGraphProjection.get, (s) => getChildRecords(s, id)),
    canComplete: (id) => Effect.map(taskGraphProjection.get, (s) => canCompleteRecord(s, id)),
    canAssign: (id, assignee) => Effect.map(taskGraphProjection.get, (s) => canAssignRecord(s, id, assignee)),
    getSubtree: (id) => Effect.map(taskGraphProjection.get, (s) => collectSubtreeRecords(s, id)),
  })

  const goalStateReaderLayer = Layer.succeed(GoalStateReaderTag, {
    getState: () => goalProjection.get,
  } satisfies GoalStateReader)

  const policyCtxProvider = createPolicyContextProvider(
    forkId,
    cwd,
    scratchpadPath,
    sessionOptions,
    agentLifecycleProjection,
    workingStateProjection,
  )

  const providedInterceptor: ToolInterceptor = {
    beforeExecute: (ctx) =>
      policyInterceptor(ctx).pipe(
        Effect.provideService(ForkContext, { forkId, roleId }),
        Effect.provideService(PolicyContextProviderTag, policyCtxProvider),
      ),
  }

  return Layer.mergeAll(
    Layer.succeed(ForkContext, { forkId, roleId }),

    agentRegistryStateReaderLayer,
    conversationStateReaderLayer,
    taskGraphReaderLayer,
    goalStateReaderLayer,
    agentStateReaderLayer,
    windowStateReaderLayer,


    Layer.succeed(WorkingDirectoryTag, { cwd, scratchpadPath }),
    Layer.succeed(PolicyContextProviderTag, policyCtxProvider),
    Layer.succeed(ToolInterceptorTag, providedInterceptor),
    persistenceLayer,

    Layer.succeed(ProjectionReaderTag, {
      getAgentRouting: () => agentProjection.get,
      getAgentState: () => agentLifecycleProjection.get,
    } satisfies ProjectionReader),

    Layer.succeed(DetachedShellRegistry, detachedShellRegistryService),
    vcsLayer,
  )
}

/**
 * Create the execution manager.
 */
const makeExecutionManager = Effect.gen(function* () {
  // Per-fork cached layers (built during initFork, reused across turns)
  const forkLayers = new Map<string | null, ForkLayer>()
  const forkCwds = new Map<string | null, string>()
  const forkScratchpadPaths = new Map<string | null, string | undefined>()

  // Bound observables map
  const boundObservables = new Map<string | null, BoundObservable[]>()

  // Approval state for gated tool calls
  // Maps forkId → roleId, populated when forks are created.
  const forkRoles = new Map<string, RoleId>()

  // Pre-built teardown effects (captured at initFork time with services already provided)
  const forkTeardowns = new Map<string, Effect.Effect<void>>()

  // Per-fork consecutive identical continue-response tracker
  const identicalContinueTracker = new Map<string | null, { lastResponseText: string; consecutiveCount: number }>()

  // Create the detached shell registry — one per execution manager lifecycle.
  // Provided in fork layers so the shell tool can access it via Effect context.
  const detachedShellRegistryService = yield* makeDetachedShellRegistryService

  // VCS layer — built once per execution manager (session scope).
  // All forks in a session share the same working directory and shadow VCS repo.
  let vcsLayer: Layer.Layer<ShadowVcs, never, never>

  /**
   * Resolve the active agent definition for a fork.
   * Child forks use their fixed role. Root fork uses the orchestrator definition.
   */
  const resolveAgent: AgentResolver = (forkId) => {
    if (forkId !== null) {
      const roleId = forkRoles.get(forkId) ?? 'engineer'
      return getAgentDefinition(roleId)
    }
    return getAgentDefinition('leader')
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
    detachedProcessCount: detachedShellRegistryService.activeCount,
    detachedProcessChanges: detachedShellRegistryService.changes,
    initFork: (forkId, roleId) => (Effect.gen(function* () {
      yield* WorkerBusTag<AppEvent>()

      const ambientService = yield* AmbientServiceTag
      const sessionOptions = ambientService.getValue(SessionOptionsAmbient)

      const sessionContextProjection = yield* SessionContextProjection.Tag
      const agentProjection = yield* AgentRoutingProjection.Tag
      const agentLifecycleProjection = yield* AgentLifecycleProjection.Tag
      const workingStateProjection = yield* TurnProjection.Tag
      const taskGraphProjection = yield* TaskGraphProjection.Tag
      const goalProjection = yield* GoalProjection.Tag
      const windowProjection = yield* WindowProjection.Tag

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
      const scratchpadPath = sessionState.context.scratchpadPath
      // Build VCS layer once per session (shared across all forks)
      if (!vcsLayer) {
        if (sessionOptions.vcsAvailable) {
          vcsLayer = makeShadowVcsLayer({
            worktreePath: cwd,
            storagePath: path.join(path.dirname(scratchpadPath), 'vcs'),
            timezone: sessionOptions.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
          }).pipe(Layer.provide(VcsFsLive))
          logger.info({ cwd, scratchpadPath }, '[ExecutionManager] VCS layer built')
        } else {
          vcsLayer = makeNoOpVcsLayer()
          logger.info({ cwd }, '[ExecutionManager] VCS disabled — cwd is home or wider, or directory exceeds size limit')
        }
      }

      let layers = makeForkLayers(
        forkId,
        roleId,
        sessionContextProjection, agentProjection, agentLifecycleProjection,
        workingStateProjection, taskGraphProjection,
        goalProjection,
        windowProjection,
        conversationProjection,
        persistenceLayer, policyInterceptor, cwd, scratchpadPath, sessionOptions,
        detachedShellRegistryService,
        vcsLayer,
      )
      forkCwds.set(forkId, cwd)
      forkScratchpadPaths.set(forkId, scratchpadPath)

      // Inject role-specific setup layer when the role defines a setup function
      const roleDef = getAgentDefinition(roleId)
      if (roleDef.setup && forkId) {
        const setupLayer = yield* roleDef.setup({ forkId, roleId, cwd, scratchpadPath })
        layers = Layer.merge(layers, setupLayer)
      }

      // Pre-build teardown effect (so disposeFork needs no requirements)
      if (forkId && roleDef.teardown) {
        const teardownEffect = roleDef.teardown({ forkId, roleId, cwd, scratchpadPath }) as Effect.Effect<void>
        forkTeardowns.set(forkId, teardownEffect)
      }

      // Store roleId for agent resolution
      if (forkId !== null) {
        forkRoles.set(forkId, roleId)
      }

      // Cache the layers
      forkLayers.set(forkId, layers)

      // Bind the detached shell registry to the event bus so it can publish events.
      // The bus is the same WorkerBusService across all forks — binding once at
      // the first initFork call (root or child) is sufficient.
      yield* detachedShellRegistryService.bindBus(
        yield* WorkerBusTag<AppEvent>()
      )

      // Bind observables
      const agentDef = getAgentDefinition(roleId)
      const agentObservables = (agentDef.observables ?? []).map((obs) =>
        bindObservable(obs, (effect) => Effect.provide(effect, layers))
      )
      boundObservables.set(forkId, agentObservables)
    }) as Effect.Effect<void, never, ExecutionProjectionRequirements | ChatPersistence | WorkerBusService<AppEvent>>),

    disposeFork: (forkId) => Effect.gen(function* () {
      // Run role teardown if defined (only for child forks)
      if (forkId !== null) {
        const teardown = forkTeardowns.get(forkId)
        if (teardown) {
          yield* Effect.ignore(teardown)
          forkTeardowns.delete(forkId)
        }
      }

      // Kill any detached processes for this fork
      yield* detachedShellRegistryService.killAll(forkId)

      forkLayers.delete(forkId)
      forkCwds.delete(forkId)
      forkScratchpadPaths.delete(forkId)

      boundObservables.delete(forkId)
      if (forkId !== null) {
        forkRoles.delete(forkId)
      }
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
      role: RoleId
      taskId: string
    }) => Effect.gen(function* () {
      const forkId = createId()
      forkRoles.set(forkId, params.role)
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
