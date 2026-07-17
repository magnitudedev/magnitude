import { AmbientServiceTag, Projection, type AmbientService, type WorkerReadFn } from '@magnitudedev/event-core'
import { Effect, Schema } from 'effect'
import type { Toolkit } from '@magnitudedev/harness'
import type { AppEvent } from '../events'
import { ConfigAmbient, type ConfigState } from '../ambient/config-ambient'
import { SessionOptionsAmbient, type SessionOptions } from '../ambient/session-ambient'
import { ToolUniverseAmbient } from '../ambient/tool-universe-ambient'
import { AgentLifecycleProjection } from './agent-lifecycle'
import { getForkInfo } from '../agents/registry'
import { selectAgentToolKeys } from '../tools/toolkits'

export const AgentToolkitStateSchema = Schema.Struct({
  configRevision: Schema.Number,
  toolKeys: Schema.Array(Schema.String),
})
export type AgentToolkitState = typeof AgentToolkitStateSchema.Type

const initial: AgentToolkitState = { configRevision: -1, toolKeys: [] }

function sameKeys(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index])
}

function select(
  roleId: Parameters<typeof selectAgentToolKeys>[0]['roleId'],
  config: ConfigState,
  options: SessionOptions,
  universe: Toolkit,
): AgentToolkitState {
  const toolKeys = selectAgentToolKeys({
    roleId,
    configState: config,
    solo: options.solo,
    vcsAvailable: options.vcsAvailable,
  })
  const missing = toolKeys.filter(key => !(key in universe.entries))
  if (missing.length > 0) throw new Error(`Tool universe is missing selected keys: ${missing.join(', ')}`)
  return {
    configRevision: config.revision,
    toolKeys,
  }
}

function recomputeAll(args: {
  readonly state: { readonly forks: ReadonlyMap<string | null, AgentToolkitState> }
  readonly agentState: Parameters<typeof getForkInfo>[0]
  readonly config: ConfigState
  readonly options: SessionOptions
  readonly universe: Toolkit
}) {
  const forks = new Map(args.state.forks)
  for (const forkId of forks.keys()) {
    const info = getForkInfo(args.agentState, forkId)
    if (!info) continue
    const next = select(info.roleId, args.config, args.options, args.universe)
    const current = forks.get(forkId)
    if (!current || current.configRevision !== next.configRevision || !sameKeys(current.toolKeys, next.toolKeys)) {
      forks.set(forkId, next)
    }
  }
  return { forks }
}

export const AgentToolkitProjection = Projection.defineForked<AppEvent>()({
  name: 'AgentToolkit',
  forkState: AgentToolkitStateSchema,
  initialFork: initial,
  reads: [AgentLifecycleProjection] as const,
  ambients: [ConfigAmbient, SessionOptionsAmbient, ToolUniverseAmbient] as const,

  eventHandlers: {
    session_initialized: ({ fork, ambient }) => select(
      'leader',
      ambient.get(ConfigAmbient),
      ambient.get(SessionOptionsAmbient),
      ambient.get(ToolUniverseAmbient),
    ),
    agent_created: ({ event, ambient }) => select(
      event.role,
      ambient.get(ConfigAmbient),
      ambient.get(SessionOptionsAmbient),
      ambient.get(ToolUniverseAmbient),
    ),
    agent_killed: () => initial,
  },

  ambientHandlers: (on) => ([
    on(ConfigAmbient, ({ value, state, read, ambient }) => recomputeAll({
      state,
      agentState: read(AgentLifecycleProjection),
      config: value,
      options: ambient.get(SessionOptionsAmbient),
      universe: ambient.get(ToolUniverseAmbient),
    })),
    on(SessionOptionsAmbient, ({ value, state, read, ambient }) => recomputeAll({
      state,
      agentState: read(AgentLifecycleProjection),
      config: ambient.get(ConfigAmbient),
      options: value,
      universe: ambient.get(ToolUniverseAmbient),
    })),
  ] as const),
})

/**
 * Read the fork projection and its source ambient at one matching revision.
 * Ambient changes are projected asynchronously, so a turn waits for that
 * projection boundary instead of combining values from different revisions.
 */
export function readCoherentAgentToolkit(
  read: WorkerReadFn<AppEvent>,
  forkId: string | null,
): Effect.Effect<{ readonly config: ConfigState; readonly toolkit: AgentToolkitState }, never, AmbientService> {
  return Effect.suspend(() => Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    const config = ambientService.getValue(ConfigAmbient)
    const toolkit = yield* read(AgentToolkitProjection, forkId)
    if (toolkit.configRevision === config.revision) return { config, toolkit }
    yield* Effect.yieldNow()
    return yield* readCoherentAgentToolkit(read, forkId)
  }))
}
