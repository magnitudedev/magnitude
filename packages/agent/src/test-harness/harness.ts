import { Agent, type Projection } from '@magnitudedev/event-core'
import { createTool, type Tool } from '@magnitudedev/tools'
import { Context, Effect, Layer } from 'effect'
import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'
import type { AppEvent, SessionContext } from '../events'
import { textParts } from '../content'


// Projections
import { SessionContextProjection } from '../projections/session-context'
import { WorkingStateProjection } from '../projections/working-state'
import { TurnProjection } from '../projections/turn'
import { CanonicalTurnProjection } from '../projections/canonical-turn'
import { MemoryProjection, getView } from '../projections/memory'
import { SubagentActivityProjection } from '../projections/subagent-activity'
import { DisplayProjection } from '../projections/display'
import { AgentRoutingProjection } from '../projections/agent-routing'
import { AgentStatusProjection } from '../projections/agent-status'
import { CompactionProjection } from '../projections/compaction'
import { ArtifactProjection } from '../projections/artifact'

import { ReplayProjection } from '../projections/replay'
import { ChatTitleProjection } from '../projections/chat-title'
import { ConversationProjection } from '../projections/conversation'
import { UserPresenceProjection } from '../projections/user-presence'
import { OutboundMessagesProjection } from '../projections/outbound-messages'
import { ArtifactAwarenessProjection } from '../projections/artifact-awareness'

// Workers
import { TurnController } from '../workers/turn-controller'
import { AgentOrchestrator } from '../workers/agent-orchestrator'
import { LifecycleCoordinator } from '../workers/lifecycle-coordinator'
import { ApprovalWorker } from '../workers/approval-worker'
import { ArtifactSyncWorker } from '../workers/artifact-sync-worker'
import { Autopilot } from '../workers/autopilot'
import { CompactionWorker } from '../workers/compaction-worker'
import { ChatTitleWorker } from '../workers/chat-title-worker'
import { UserPresenceWorker } from '../workers/user-presence-worker'

// Runtime/services
import { ExecutionManager, ExecutionManagerLive } from '../execution/execution-manager'
import { BrowserServiceLive } from '../services/browser-service'
import { makeProviderRuntimeLive, makeTestResolver, type TestModelConfig } from '@magnitudedev/providers'
import { registerApprovalBridge } from '../execution/approval-bridge'

// Testing services
import { InMemoryChatPersistenceTag, makeInMemoryChatPersistenceLayer } from './in-memory-persistence'
import { MockCortex } from './mock-cortex'
import { MockTurnScriptTag, MockTurnScriptLive, createScriptGate, type MockTurnResponse, type MockTurnScriptResolver, type ScriptGate } from './turn-script'
import { response as standaloneResponse } from './response-builder'
import { createTurnsBuilder } from './scenario-builder'
import { clearAgentOverrides, getAgentDefinition, registerAgentDefinition, type AgentVariant } from '../agents'
import { defaultXmlTagName } from '../tools'
import { createDefaultToolOverrides, createVirtualFs } from './virtual-fs'
import { EphemeralSessionContextTag, type PolicyContext } from '../agents/types'
import { ChatPersistence, PersistenceError, type ChatPersistenceService } from '../persistence/chat-persistence-service'
import { createFaultRegistry, type FaultPlan, type FaultRegistry, type FaultScope } from './faults'
import { createFakeClock } from './fake-clock'

export type ToolOverrideHandler = (input: unknown) => unknown | Promise<unknown>

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
  artifacts: Record<string, unknown>
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
  }
  defaults?: {
    waitTimeoutMs?: number
  }
  workers?: {
    autopilot?: boolean
    compaction?: boolean
    chatTitle?: boolean
    userPresence?: boolean
  }
  files?: Record<string, string>
  toolOverrides?: Record<string, ToolOverrideHandler>
  extraLayers?: Layer.Layer<unknown, never, never>[]
  clock?: 'real' | 'fake'
  model?: TestModelConfig
}

const ALL_VARIANTS: AgentVariant[] = [
  'orchestrator',
  'builder',
  'explorer',
  'planner',
  'debugger',
  'reviewer',
  'browser',
]

type MagnitudeAgentDef = AgentDefinition<ToolSet, PolicyContext>

function makeOverrideTool(source: Tool.Any, handler: ToolOverrideHandler): Tool.Any {
  const execute: Tool.Any['execute'] = (input) =>
    Effect.tryPromise({
      try: () => Promise.resolve(handler(input)),
      catch: (error) => (error instanceof Error ? error : new Error(String(error))),
    }).pipe(Effect.orDie)

  return {
    ...source,
    execute,
  }
}

function applyToolOverrides(
  handlers: Record<string, ToolOverrideHandler>,
  faultRegistry?: FaultRegistry,
): void {
  for (const variant of ALL_VARIANTS) {
    const def = getAgentDefinition(variant)
    const tools: Partial<Record<keyof typeof def.tools, Tool.Any>> = {}

    for (const key of Object.keys(def.tools) as Array<keyof typeof def.tools>) {
      const concreteTool = def.tools[key]
      if (!concreteTool) {
        continue
      }
      const tagName = defaultXmlTagName(concreteTool)
      const override = handlers[tagName]
      const wrappedOverride = override
        ? async (input: unknown) => {
          await faultRegistry?.checkAsync(`tool.execute:${tagName}`)
          return override(input)
        }
        : null

      tools[key] = wrappedOverride
        ? makeOverrideTool(concreteTool, wrappedOverride)
        : concreteTool
    }

    const overridden: MagnitudeAgentDef = {
      ...def,
      tools: tools as MagnitudeAgentDef['tools'],
    }

    registerAgentDefinition(variant, overridden)
  }
}

const DEFAULT_TIMEOUT_MS = 10_000

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
    userMemory: null,
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
  const defaultOverrides = createDefaultToolOverrides(files)
  const handlers: Record<string, ToolOverrideHandler> = {
    ...defaultOverrides,
    ...(options.toolOverrides ?? {}),
  }
  applyToolOverrides(handlers, faultRegistry)

  try {
    const workers = [
      TurnController,
      MockCortex,
      AgentOrchestrator,
      LifecycleCoordinator,
      ApprovalWorker,
      ArtifactSyncWorker,
      ...(options.workers?.autopilot ? [Autopilot] : []),
      ...(options.workers?.compaction ? [CompactionWorker] : []),
      ...(options.workers?.chatTitle ? [ChatTitleWorker] : []),
      ...(options.workers?.userPresence ? [UserPresenceWorker] : []),
    ] as const

    const TestCodingAgent = Agent.define<AppEvent>()({
      name: 'TestCodingAgent',
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
      workers,
      expose: {
        signals: {
          restoreQueuedMessages: DisplayProjection.signals.restoreQueuedMessages,
          chatTitleGenerated: ChatTitleProjection.signals.chatTitleGenerated,
        },
        state: {
          display: DisplayProjection,
          working: WorkingStateProjection,
          memory: MemoryProjection,
          compaction: CompactionProjection,
          agentRouting: AgentRoutingProjection,
          agentStatus: AgentStatusProjection,
          artifacts: ArtifactProjection,

        },
      },
    })

    const basePersistenceLayer = makeInMemoryChatPersistenceLayer({
      events: options.persistence?.seedEvents ?? [],
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

    const providerRuntime = makeProviderRuntimeLive()
    const ephemeralSessionContextLayer = Layer.succeed(EphemeralSessionContextTag, {
      disableShellSafeguards: false,
      disableCwdSafeguards: false,
    })
    const runtimeLayer = Layer.mergeAll(
      Layer.provide(ExecutionManagerLive, ephemeralSessionContextLayer),
      Layer.provide(BrowserServiceLive, providerRuntime),
      providerRuntime,
      ...(options.model
        ? [makeTestResolver(options.model as TestModelConfig)]
        : [makeTestResolver()]),
      MockTurnScriptLive,
      basePersistenceLayer,
      faultWrappedPersistenceLayer,
      ...(fakeClock ? [fakeClock.layer] : []),
      ...(options.extraLayers ?? []),
    )
    const client = await TestCodingAgent.createClient(runtimeLayer)

    await client.runEffect(registerApprovalBridge)
    await client.runEffect(
      Effect.flatMap(ExecutionManager, (manager) => manager.initFork(null, 'orchestrator'))
    )

    const transcript: AppEvent[] = []
    const listeners = new Set<(e: AppEvent) => void>()
    const scriptGates = new Map<string, ScriptGate>()
    const unsubscribeClient = client.onEvent((event) => {
      transcript.push(event)
      for (const listener of listeners) listener(event)
    })

    const send = async (event: AppEvent): Promise<void> => {
      await client.send(event)
    }

    await send({
      type: 'session_initialized',
      forkId: null,
      context: defaultSessionContext(options.sessionContext),
    })

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
        await send({
          type: 'user_message',
          forkId: null,
          content: textParts(text),
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
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
            if (!entry || !mapping.subagents) return { xml: '<yield/>' }

            if ('xml' in mapping.subagents || 'xmlChunks' in mapping.subagents) {
              return mapping.subagents
            }

            const [agentId] = entry
            const perAgent = mapping.subagents as Record<string, MockTurnResponse>
            return perAgent[agentId] ?? { xml: '<yield/>' }
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
            artifact,
            compaction,
            working,
            memory,
            agentRouting,
            agentStatus,
            sessionContext,
          ] = await Promise.all([
            client.runEffect(Effect.flatMap(ArtifactProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(CompactionProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(WorkingStateProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(MemoryProjection.Tag, (projection) => projection.getFork(null))),
            client.runEffect(Effect.flatMap(AgentRoutingProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(AgentStatusProjection.Tag, (projection) => projection.get)),
            client.runEffect(Effect.flatMap(SessionContextProjection.Tag, (projection) => projection.get)),
          ])

          return {
            ArtifactProjection: artifact,
            CompactionProjection: compaction,
            WorkingStateProjection: working,
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
            artifacts: { ...state.artifacts },
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
          const working = await client.runEffect(
            Effect.flatMap(WorkingStateProjection.Tag, (projection) => projection.getFork(forkId))
          )
          if (working.contextLimitBlocked !== false) {
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
          const workingState = await client.runEffect(
            Effect.flatMap(WorkingStateProjection.Tag, (projection) => projection.get)
          )

          for (const [forkId, working] of workingState.forks) {
            if (working.working || working.willContinue) {
              await send({ type: 'interrupt', forkId } as AppEvent)
            }
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