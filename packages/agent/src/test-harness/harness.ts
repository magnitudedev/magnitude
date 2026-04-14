import { Agent, type Projection, makeAmbientServiceLayer } from '@magnitudedev/event-core'
import { defineCatalog } from '@magnitudedev/tools'
import { Context, Effect, Layer, SubscriptionRef } from 'effect'
import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'
import type { RoleDefinition } from '@magnitudedev/roles'
import type { AgentCatalogEntry } from '../catalog'
import type { AppEvent, SessionContext } from '../events'
import { textParts } from '../content'
import { createId } from '../util/id'


// Projections
import { SessionContextProjection } from '../projections/session-context'
import { TaskGraphProjection } from '../projections/task-graph'
import { TurnProjection } from '../projections/turn'
import { CanonicalTurnProjection } from '../projections/canonical-turn'
import { MemoryProjection, getView } from '../projections/memory'
import { SubagentActivityProjection } from '../projections/subagent-activity'
import { DisplayProjection } from '../projections/display'
import { ToolStateProjection } from '../projections/tool-state'
import { TaskWorkerProjection } from '../projections/task-worker'
import { AgentRoutingProjection } from '../projections/agent-routing'
import { AgentStatusProjection } from '../projections/agent-status'
import { CompactionProjection } from '../projections/compaction'

import { ReplayProjection } from '../projections/replay'
import { ConversationProjection } from '../projections/conversation'
import { UserPresenceProjection } from '../projections/user-presence'
import { OutboundMessagesProjection } from '../projections/outbound-messages'
import { UserMessageResolutionProjection } from '../projections/user-message-resolution'


// Workers
import { TurnController } from '../workers/turn-controller'
import { AgentLifecycle } from '../workers/agent-lifecycle'
import { LifecycleCoordinator } from '../workers/lifecycle-coordinator'
import { ApprovalWorker } from '../workers/approval-worker'
import { Autopilot } from '../workers/autopilot'
import { CompactionWorker } from '../workers/compaction-worker'
import { UserPresenceWorker } from '../workers/user-presence-worker'
import { FileMentionResolver } from '../workers/file-mention-resolver'
import { SessionTitleWorker } from '../workers/session-title-worker'

// Runtime/services
import { ExecutionManagerLive } from '../execution/execution-manager'
import { BrowserServiceLive } from '../services/browser-service'
import { ModelCatalog, makeProviderRuntimeLive, makeTestResolver, type TestModelConfig } from '@magnitudedev/providers'
import type { MagnitudeSlot } from '../model-slots'
import { registerApprovalBridge } from '../execution/approval-bridge'

// Testing services
import { InMemoryChatPersistenceTag, makeInMemoryChatPersistenceLayer } from './in-memory-persistence'
import { MockCortex } from './mock-cortex'
import { MockTurnScriptTag, MockTurnScriptLive, createScriptGate, type MockTurnResponse, type MockTurnScriptResolver, type ScriptGate } from './turn-script'
import { response as standaloneResponse } from './response-builder'
import { createTurnsBuilder } from './scenario-builder'
import { clearAgentOverrides } from '../agents'
import { createVirtualFs, createVirtualFsLayer } from './virtual-fs'
import { EphemeralSessionContextTag, type PolicyContext } from '../agents/types'
import { SkillsetResolverLive } from '@magnitudedev/skills'
import { ChatPersistence, PersistenceError, type ChatPersistenceService } from '../persistence/chat-persistence-service'
import { createFaultRegistry, type FaultPlan, type FaultRegistry, type FaultScope } from './faults'
import { createFakeClock } from './fake-clock'

export interface WaitOptions {
  timeoutMs?: number
}

export interface WaitUntilOptions extends WaitOptions {
  pollIntervalMs?: number
}

export interface HarnessSnapshot {
  eventCount: number
  projections: Record<string, unknown>
}

export interface PersistenceSnapshot {
  events: readonly AppEvent[]
  metadata: Record<string, unknown>
}

export interface ContextSnapshot {
  messages: Array<{ role: string; content: string }>
  systemPrompt?: string
  tokenEstimate?: number
}

export interface HarnessOptions {
  sessionContext?: Partial<SessionContext>
  persistence?: {
    seedEvents?: AppEvent[]
    metadata?: {
      chatName?: string
      workingDirectory?: string
      gitBranch?: string | null
    }
  }
  defaults?: {
    waitTimeoutMs?: number
  }
  workers?: {
    turnController?: boolean
    autopilot?: boolean
    compaction?: boolean
    userPresence?: boolean
  }
  files?: Record<string, string>
  extraLayers?: Layer.Layer<unknown, never, never>[]
  clock?: 'real' | 'fake'
  model?: TestModelConfig
}

type MagnitudeAgentDef = RoleDefinition<import('../catalog').AgentCatalog, MagnitudeSlot, PolicyContext>

const DEFAULT_TIMEOUT_MS = 10_000

export interface TestHarnessService {
  readonly client: AgentTestHarness['client']
  readonly files: AgentTestHarness['files']
  readonly send: (event: AppEvent) => Effect.Effect<void>
  readonly user: (text: string) => Effect.Effect<void>
  readonly events: () => readonly AppEvent[]
  readonly wait: {
    readonly event: <T extends AppEvent['type']>(
      type: T,
      pred?: (e: Extract<AppEvent, { type: T }>) => boolean,
      opts?: WaitOptions,
    ) => Effect.Effect<Extract<AppEvent, { type: T }>>
    readonly turnCompleted: (forkId?: string | null, opts?: WaitOptions) => Effect.Effect<Extract<AppEvent, { type: 'turn_completed' }>>
    readonly idle: (forkId?: string | null, opts?: WaitOptions) => Effect.Effect<void>
    readonly agentCreated: (
      pred?: (e: Extract<AppEvent, { type: 'agent_created' }>) => boolean,
      opts?: WaitOptions,
    ) => Effect.Effect<Extract<AppEvent, { type: 'agent_created' }>>
  }
  readonly script: {
    readonly next: (frame: MockTurnResponse, forkId?: string | null) => Effect.Effect<void>
    readonly setResolver: (resolver: MockTurnScriptResolver | null) => Effect.Effect<void>
    readonly route: (mapping: ScriptRouteMapping) => Effect.Effect<() => void>
  }
  readonly projectionFork: <S>(
    tag: Context.Tag<Projection.ForkedProjectionInstance<S>, Projection.ForkedProjectionInstance<S>>,
    forkId: string | null,
  ) => Effect.Effect<S>
  readonly projection: <S>(
    tag: Context.Tag<Projection.ProjectionInstance<S>, Projection.ProjectionInstance<S>>,
  ) => Effect.Effect<S>
  readonly runEffect: <A, E>(effect: Effect.Effect<A, E, any>) => Effect.Effect<A, E>
}

export class TestHarness extends Context.Tag('TestHarness')<TestHarness, TestHarnessService>() {}

export function TestHarnessLive(options: HarnessOptions = {}): Layer.Layer<TestHarness> {
  return Layer.scoped(
    TestHarness,
    Effect.acquireRelease(
      Effect.promise(() => createAgentTestHarness(options)),
      (harness) => Effect.promise(() => harness.dispose()),
    ).pipe(
      Effect.map((harness): TestHarnessService => ({
        client: harness.client,
        files: harness.files,
        send: (event) => Effect.promise(() => harness.send(event)),
        user: (text) => Effect.promise(() => harness.user(text)),
        events: () => harness.events(),
        wait: {
          event: (type, pred, opts) => Effect.promise(() => harness.wait.event(type, pred as never, opts)),
          turnCompleted: (forkId = null, opts) => Effect.promise(() => harness.wait.turnCompleted(forkId, opts)),
          idle: (forkId = null, opts) => Effect.promise(() => harness.wait.idle(forkId, opts)),
          agentCreated: (pred, opts) => Effect.promise(() => harness.wait.agentCreated(pred, opts)),
        },
        script: {
          next: (frame, forkId = null) => Effect.promise(() => harness.script.next(frame, forkId)),
          setResolver: (resolver) => Effect.promise(() => harness.script.setResolver(resolver)),
          route: (mapping) => Effect.promise(() => harness.script.route(mapping)),
        },
        projectionFork: (tag, forkId) => Effect.promise(() => harness.projectionFork(tag, forkId)),
        projection: (tag) => Effect.promise(() => harness.projection(tag)),
        runEffect: (effect) => Effect.promise(() => harness.runEffect(effect)),
      })),
    ),
  )
}

function defaultSessionContext(overrides: Partial<SessionContext> = {}): SessionContext {
  return {
    cwd: process.cwd(),
    platform: process.platform === 'darwin' ? 'macos' : process.platform === 'win32' ? 'windows' : 'linux',
    shell: process.env.SHELL ?? '/bin/zsh',
    timezone: 'UTC',
    username: process.env.USER ?? 'tester',
    workspacePath: '/tmp/test-workspace',
    fullName: null,
    git: null,
    folderStructure: '.',
    agentsFile: null,
    skills: null,
    ...overrides,
  }
}

type ScriptRouteMapping = {
  readonly root: MockTurnResponse
  readonly subagents?: MockTurnResponse | Record<string, MockTurnResponse>
}

export async function createAgentTestHarness(options: HarnessOptions = {}) {
  clearAgentOverrides()
  const files = createVirtualFs(options.files)
  const faultRegistry = createFaultRegistry()

  try {
    const workers = [
      ...(options.workers?.turnController !== false ? [TurnController] : []),
      MockCortex,
      AgentLifecycle,
      LifecycleCoordinator,
      ApprovalWorker,
      ...(options.workers?.autopilot ? [Autopilot] : []),
      ...(options.workers?.compaction ? [CompactionWorker] : []),
      ...(options.workers?.userPresence ? [UserPresenceWorker] : []),

      FileMentionResolver,
      SessionTitleWorker,
    ] as const

    const TestCodingAgent = Agent.define<AppEvent>()({
      name: 'TestCodingAgent',
      projections: [
        SessionContextProjection,
        AgentRoutingProjection,
        AgentStatusProjection,
        CompactionProjection,
        TaskGraphProjection,
        TurnProjection,
        CanonicalTurnProjection,

        ReplayProjection,
        SubagentActivityProjection,
        OutboundMessagesProjection,
        UserMessageResolutionProjection,
        ToolStateProjection,
        TaskWorkerProjection,
        MemoryProjection,
        DisplayProjection,
        ConversationProjection,
        UserPresenceProjection,
      ],
      workers,
      expose: {
        signals: {
          restoreQueuedMessages: DisplayProjection.signals.restoreQueuedMessages,
        },
        state: {
          display: DisplayProjection,
          toolState: ToolStateProjection,
          taskWorker: TaskWorkerProjection,
          turn: TurnProjection,
          memory: MemoryProjection,
          compaction: CompactionProjection,
          agentRouting: AgentRoutingProjection,
          agentStatus: AgentStatusProjection,

        },
      },
    })

    const basePersistenceLayer = makeInMemoryChatPersistenceLayer({
      events: options.persistence?.seedEvents ?? [],
      metadata: options.persistence?.metadata,
    })

    const faultWrappedPersistenceLayer = Layer.effect(
      ChatPersistence,
      Effect.map(InMemoryChatPersistenceTag, (persistence): ChatPersistenceService => ({
        ...persistence,
        loadEvents: () =>
          Effect.tryPromise({
            try: async () => {
              await faultRegistry.checkAsync('persistence.loadEvents')
            },
            catch: (error) => new PersistenceError({ reason: 'BackendError', message: error instanceof Error ? error.message : String(error) }),
          }).pipe(
            Effect.flatMap(() => persistence.loadEvents())
          ),
        persistNewEvents: (events) =>
          Effect.tryPromise({
            try: async () => {
              await faultRegistry.checkAsync('persistence.persistNewEvents')
            },
            catch: (error) => new PersistenceError({ reason: 'BackendError', message: error instanceof Error ? error.message : String(error) }),
          }).pipe(
            Effect.flatMap(() => persistence.persistNewEvents(events))
          ),
      }))
    ).pipe(Layer.provide(basePersistenceLayer))

    const defaultWaitTimeoutMs = options.defaults?.waitTimeoutMs ?? DEFAULT_TIMEOUT_MS
    const fakeClock = options.clock === 'fake' ? createFakeClock() : null

    const testModelCatalogLayer = Layer.succeed(ModelCatalog, {
      refresh: () => Effect.void,
      getModels: () => Effect.succeed([]),
    })
    const providerRuntime = makeProviderRuntimeLive<MagnitudeSlot>(testModelCatalogLayer)
    const ephemeralSessionContextLayer = Layer.succeed(EphemeralSessionContextTag, {
      disableShellSafeguards: false,
      disableCwdSafeguards: false,
    })
    const fsLayer = createVirtualFsLayer(
      files,
      options.sessionContext?.cwd ?? process.cwd(),
      options.sessionContext?.workspacePath ?? '/tmp/test-workspace',
    )

    const runtimeLayer = Layer.mergeAll(
      Layer.provide(ExecutionManagerLive, ephemeralSessionContextLayer),
      Layer.provide(BrowserServiceLive, providerRuntime),
      providerRuntime,
      fsLayer,
      ...(options.model
        ? [makeTestResolver(options.model as TestModelConfig)]
        : [makeTestResolver()]),
      MockTurnScriptLive,
      basePersistenceLayer,
      faultWrappedPersistenceLayer,
      SkillsetResolverLive,
      ...(fakeClock ? [fakeClock.layer] : []),
      ...(options.extraLayers ?? []),
    )
    const client = await TestCodingAgent.createClient(runtimeLayer)

    await client.runEffect(registerApprovalBridge)

    const transcript: AppEvent[] = []
    const listeners = new Set<(e: AppEvent) => void>()
    const scriptGates = new Map<string, ScriptGate>()
    const unsubscribeClient = client.onEvent((event) => {
      transcript.push(event)
      for (const listener of listeners) listener(event)
    })

    await client.send({
      type: 'session_initialized',
      forkId: null,
      context: defaultSessionContext(options.sessionContext),
    } as AppEvent)

    // Let AgentLifecycle process session initialization and initialize root fork
    await new Promise((resolve) => setTimeout(resolve, 100))

    const send = async (event: AppEvent): Promise<void> => {
      await client.send(event)
    }

    const onEvent = (cb: (e: AppEvent) => void): (() => void) => {
      listeners.add(cb)
      return () => { listeners.delete(cb) }
    }

    const waitEvent = <T extends AppEvent['type']>(
      type: T,
      pred?: (e: Extract<AppEvent, { type: T }>) => boolean,
      opts?: WaitOptions,
    ): Promise<Extract<AppEvent, { type: T }>> =>
      new Promise((resolve, reject) => {
        for (const ev of transcript) {
          if (ev.type === type) {
            const match = ev as Extract<AppEvent, { type: T }>
            if (!pred || pred(match)) {
              resolve(match)
              return
            }
          }
        }

        const timeout = setTimeout(() => {
          unsub()
          reject(new Error(`Timed out waiting for event "${type}"`))
        }, opts?.timeoutMs ?? defaultWaitTimeoutMs)

        const unsub = onEvent((ev) => {
          if (ev.type !== type) return
          const match = ev as Extract<AppEvent, { type: T }>
          if (pred && !pred(match)) return
          clearTimeout(timeout)
          unsub()
          resolve(match)
        })
      })

    const waitUntil = async (
      label: string,
      pred: () => boolean | Promise<boolean>,
      opts?: WaitUntilOptions,
    ): Promise<void> => {
      const timeoutMs = opts?.timeoutMs ?? defaultWaitTimeoutMs
      const pollIntervalMs = opts?.pollIntervalMs ?? 50
      const start = Date.now()

      while (true) {
        if (await pred()) {
          return
        }
        if (Date.now() - start >= timeoutMs) {
          throw new Error(`Timed out waiting for condition "${label}"`)
        }
        await new Promise<void>((resolve) => setTimeout(resolve, pollIntervalMs))
      }
    }

    const builtHarness = {
      client,
      files,
      send,
      user: async (text: string) => {
        const timestamp = Date.now()
        const content = textParts(text)
        const messageId = createId()
        await send({
          type: 'user_message',
          messageId,
          forkId: null,
          timestamp,
          content,
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
        await waitEvent('user_message_ready', (e) => e.messageId === messageId)
      },
      events: () => transcript,
      onEvent,
      wait: {
        event: waitEvent,
        turnCompleted: (forkId: string | null = null, opts?: WaitOptions) =>
          waitEvent('turn_completed', (e) => e.forkId === forkId, opts),
        idle: async (forkId: string | null = null, opts?: WaitOptions) => {
          await waitEvent('turn_completed', (e) => e.forkId === forkId, opts)
        },
        agentCreated: (
          pred?: (e: Extract<AppEvent, { type: 'agent_created' }>) => boolean,
          opts?: WaitOptions,
        ) => waitEvent('agent_created', pred, opts),
        until: waitUntil,
      },
      script: {
        next: async (frame: MockTurnResponse, forkId: string | null = null) => {
          await client.runEffect(
            Effect.flatMap(MockTurnScriptTag, (script) => script.enqueue(frame, forkId))
          )
        },
        gate: (name: string): ScriptGate => {
          const existing = scriptGates.get(name)
          if (existing) {
            return existing
          }
          const gate = createScriptGate()
          scriptGates.set(name, gate)
          return gate
        },
        setResolver: async (resolver: MockTurnScriptResolver | null) => {
          await client.runEffect(
            Effect.flatMap(MockTurnScriptTag, (script) => script.setResolver(resolver))
          )
        },
        route: async (mapping: ScriptRouteMapping) => {
          const forksByAgent = new Map<string, string>()
          const unsub = onEvent((event) => {
            if (event.type === 'agent_created') {
              forksByAgent.set(event.agentId, event.forkId)
            }
          })

          const resolver: MockTurnScriptResolver = ({ forkId }) => {
            if (forkId === null) return mapping.root
            const entry = Array.from(forksByAgent.entries()).find(([, id]) => id === forkId)
            if (!entry || !mapping.subagents) return { xml: TURN_CONTROL_IDLE }

            if ('xml' in mapping.subagents || 'xmlChunks' in mapping.subagents) {
              return mapping.subagents
            }

            const [agentId] = entry
            const perAgent = mapping.subagents as Record<string, MockTurnResponse>
            return perAgent[agentId] ?? { xml: TURN_CONTROL_IDLE }
          }

          await client.runEffect(
            Effect.flatMap(MockTurnScriptTag, (script) => script.setResolver(resolver))
          )

          return () => {
            unsub()
          }
        },
      },
      state: client.state,
      projection: async <S>(
        tag: Context.Tag<Projection.ProjectionInstance<S>, Projection.ProjectionInstance<S>>,
      ): Promise<S> =>
        client.runEffect(
          Effect.flatMap(tag, (projection) => projection.get),
        ),
      projectionFork: async <S>(
        tag: Context.Tag<Projection.ForkedProjectionInstance<S>, Projection.ForkedProjectionInstance<S>>,
        forkId: string | null,
      ): Promise<S> =>
        client.runEffect(
          Effect.flatMap(tag, (projection) => projection.getFork(forkId)),
        ),
      response: () => standaloneResponse(),
      turns: () => createTurnsBuilder(builtHarness),
      inspect: {
        context: async (forkId: string | null = null): Promise<ContextSnapshot> => {
          const [memory, sessionContext, compaction] = await Promise.all([
            client.runEffect(Effect.flatMap(MemoryProjection.Tag, (projection) => projection.getFork(forkId))),
            client.runEffect(Effect.flatMap(SessionContextProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(CompactionProjection.Tag, (projection) => projection.getFork(forkId))),
          ])

          const timezone = sessionContext.context?.timezone ?? null
          const messages = getView(memory.messages, timezone, 'agent').map((message) => ({
            role: message.role,
            content: message.content
              .map((part) => (part.type === 'text' ? part.text : '[image]'))
              .join(''),
          }))

          return {
            messages,
            tokenEstimate: compaction.tokenEstimate,
          }
        },
        projections: async (): Promise<Record<string, unknown>> => {
          const [
            compaction,
            turn,
            memory,
            agentRouting,
            agentStatus,
            sessionContext,
          ] = await Promise.all([
            client.runEffect(Effect.flatMap(CompactionProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(TurnProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(MemoryProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(AgentRoutingProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(AgentStatusProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(SessionContextProjection.Tag, (projection) => projection.get)),
          ])

          return {
            CompactionProjection: compaction,
            TurnProjection: turn,
            MemoryProjection: memory,
            AgentRoutingProjection: agentRouting,
            AgentStatusProjection: agentStatus,
            SessionContextProjection: sessionContext,
          }
        },
        snapshot: async (): Promise<HarnessSnapshot> => ({
          eventCount: transcript.length,
          projections: await builtHarness.inspect.projections(),
        }),
        persistence: async (): Promise<PersistenceSnapshot> => {
          const state = await client.runEffect(
            Effect.flatMap(InMemoryChatPersistenceTag, (persistence) => persistence.inspectState())
          )
          return {
            events: state.events,
            metadata: { ...state.metadata },
          }
        },
      },
      compaction: {
        trigger: async (forkId: string | null = null): Promise<void> => {
          await send({
            type: 'context_limit_hit',
            forkId,
            error: 'forced by harness compaction.trigger',
          })
        },
        waitReady: (forkId: string | null = null) =>
          waitEvent('compaction_ready', (e) => e.forkId === forkId),
        waitCompleted: (forkId: string | null = null) =>
          waitEvent('compaction_completed', (e) => e.forkId === forkId),
        assertNotBlocked: async (forkId: string | null = null): Promise<void> => {
          const compaction = await client.runEffect(
            Effect.flatMap(CompactionProjection.Tag, (projection) => projection.getFork(forkId))
          )
          if (compaction.contextLimitBlocked !== false) {
            throw new Error(`Expected contextLimitBlocked to be false for fork ${forkId ?? 'root'}`)
          }
        },
      },
      approvals: {
        approve: async (toolCallId: string): Promise<void> => {
          await send({ type: 'tool_approved', forkId: null, toolCallId })
        },
        reject: async (toolCallId: string, reason?: string): Promise<void> => {
          await send({ type: 'tool_rejected', forkId: null, toolCallId, ...(reason ? { reason } : {}) })
        },
        waitPending: () =>
          waitEvent('tool_event', (e) => {
            if (e.event._tag !== 'ToolExecutionEnded') return false
            const result = e.event.result
            if (result._tag !== 'Rejected') return false
            const rejection = result.rejection
            return typeof rejection === 'object' && rejection !== null && '_tag' in rejection && rejection._tag === 'ApprovalPending'
          }),
      },
      faults: {
        set: (plan: FaultPlan): void => {
          faultRegistry.set(plan)
        },
        clear: (scope?: FaultScope): void => {
          faultRegistry.clear(scope)
        },
      },
      clock: {
        isFake: fakeClock !== null,
        now: () => (fakeClock ? fakeClock.now() : Date.now()),
        advanceBy: async (ms: number): Promise<void> => {
          if (fakeClock) {
            await fakeClock.advanceBy(ms)
            return
          }
          if (ms > 0) {
            await new Promise<void>((resolve) => setTimeout(resolve, ms))
          }
        },
        runAll: async (): Promise<void> => {
          if (fakeClock) {
            await fakeClock.runAll()
          }
        },
      },
      replay: {
        hydrate: async (events: AppEvent[]): Promise<void> => {
          await client.runEffect(
            Effect.flatMap(InMemoryChatPersistenceTag, (persistence) => persistence.persistNewEvents(events))
          )
        },
      },
      runEffect: <A, E, R>(effect: Effect.Effect<A, E, R>) => client.runEffect(effect),
      dispose: async () => {
        unsubscribeClient()
        try {
          const turnState = await client.runEffect(
            Effect.flatMap(TurnProjection.Tag, (projection) => projection.getFork(null))
          )

          if (turnState._tag !== 'idle' || turnState.triggers.length > 0) {
            await Promise.race([
              send({ type: 'interrupt', forkId: null } as AppEvent),
              new Promise<void>((resolve) => setTimeout(resolve, 250)),
            ])
          }
        } catch {
          // ignore best-effort teardown interrupt failures
        }
        await client.dispose()
        clearAgentOverrides()
      },
    }

    return builtHarness
  } catch (error) {
    clearAgentOverrides()
    throw error
  }
}

export type AgentTestHarness = Awaited<ReturnType<typeof createAgentTestHarness>>

export async function withHarness(
  fn: (h: AgentTestHarness) => Promise<void>,
): Promise<void>
export async function withHarness(
  opts: HarnessOptions,
  fn: (h: AgentTestHarness) => Promise<void>,
): Promise<void>
export async function withHarness(
  optsOrFn: HarnessOptions | ((h: AgentTestHarness) => Promise<void>),
  maybeFn?: (h: AgentTestHarness) => Promise<void>,
): Promise<void> {
  const opts = typeof optsOrFn === 'function' ? {} : optsOrFn
  const fn = typeof optsOrFn === 'function' ? optsOrFn : maybeFn
  if (!fn) {
    throw new Error('withHarness requires a callback')
  }

  const harness = await createAgentTestHarness(opts)
  try {
    await fn(harness)
  } finally {
    await harness.dispose()
  }
}