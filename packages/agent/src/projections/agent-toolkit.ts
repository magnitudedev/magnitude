import { Projection } from '@magnitudedev/event-core'
import { Effect, Option, Schema, Stream } from 'effect'
import type { Toolkit } from '@magnitudedev/harness'
import type { AppEvent } from '../events'
import {
  ConfigAmbient,
  ConfigStateSchema,
  getSlotStateForRole,
  sameConfigStateValue,
  type ConfigState,
} from '../ambient/config-ambient'
import { SessionOptionsAmbient, type SessionOptions } from '../ambient/session-ambient'
import { ToolUniverseAmbient } from '../ambient/tool-universe-ambient'
import {
  ToolAvailabilityAmbient,
  type ToolAvailabilityState,
} from '../ambient/tool-availability-ambient'
import { AgentLifecycleProjection } from './agent-lifecycle'
import { getForkInfo } from '../agents/registry'
import { selectAgentToolKeys } from '../tools/toolkits'
import type { RoleId } from '../agents/role-validation'

export const AgentToolkitStateSchema = Schema.Struct({
  config: Schema.NullOr(ConfigStateSchema),
  toolKeys: Schema.Array(Schema.String),
})
export type AgentToolkitState = typeof AgentToolkitStateSchema.Type

const initial: AgentToolkitState = {
  config: null,
  toolKeys: [],
}

function sameKeys(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index])
}

function select(
  roleId: Parameters<typeof selectAgentToolKeys>[0]['roleId'],
  config: ConfigState,
  toolAvailability: ToolAvailabilityState,
  options: SessionOptions,
  universe: Toolkit,
): AgentToolkitState {
  const toolKeys = selectAgentToolKeys({
    roleId,
    configState: config,
    toolAvailability,
    solo: options.solo,
    vcsAvailable: options.vcsAvailable,
  })
  const missing = toolKeys.filter(key => !(key in universe.entries))
  if (missing.length > 0) throw new Error(`Tool universe is missing selected keys: ${missing.join(', ')}`)
  return {
    config,
    toolKeys,
  }
}

function recomputeAll(args: {
  readonly state: { readonly forks: ReadonlyMap<string | null, AgentToolkitState> }
  readonly agentState: Parameters<typeof getForkInfo>[0]
  readonly config: ConfigState
  readonly toolAvailability: ToolAvailabilityState
  readonly options: SessionOptions
  readonly universe: Toolkit
}) {
  const forks = new Map(args.state.forks)
  for (const forkId of forks.keys()) {
    const info = getForkInfo(args.agentState, forkId)
    if (!info) continue
    const next = select(
      info.roleId,
      args.config,
      args.toolAvailability,
      args.options,
      args.universe,
    )
    const current = forks.get(forkId)
    if (
      !current
      || current.config === null
      || !sameConfigStateValue(current.config, args.config)
      || !sameKeys(current.toolKeys, next.toolKeys)
    ) {
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
  ambients: [
    ConfigAmbient,
    ToolAvailabilityAmbient,
    SessionOptionsAmbient,
    ToolUniverseAmbient,
  ] as const,

  eventHandlers: {
    session_initialized: ({ fork, ambient }) => select(
      'leader',
      ambient.get(ConfigAmbient),
      ambient.get(ToolAvailabilityAmbient),
      ambient.get(SessionOptionsAmbient),
      ambient.get(ToolUniverseAmbient),
    ),
    agent_created: ({ event, ambient }) => select(
      event.role,
      ambient.get(ConfigAmbient),
      ambient.get(ToolAvailabilityAmbient),
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
      toolAvailability: ambient.get(ToolAvailabilityAmbient),
      options: ambient.get(SessionOptionsAmbient),
      universe: ambient.get(ToolUniverseAmbient),
    })),
    on(ToolAvailabilityAmbient, ({ value, state, read, ambient }) => recomputeAll({
      state,
      agentState: read(AgentLifecycleProjection),
      config: ambient.get(ConfigAmbient),
      toolAvailability: value,
      options: ambient.get(SessionOptionsAmbient),
      universe: ambient.get(ToolUniverseAmbient),
    })),
    on(SessionOptionsAmbient, ({ value, state, read, ambient }) => recomputeAll({
      state,
      agentState: read(AgentLifecycleProjection),
      config: ambient.get(ConfigAmbient),
      toolAvailability: ambient.get(ToolAvailabilityAmbient),
      options: value,
      universe: ambient.get(ToolUniverseAmbient),
    })),
  ] as const),
})

export function awaitSettledAgentToolkit(
  forkId: string | null,
  roleId: RoleId,
) {
  return Effect.gen(function* () {
    const projection = yield* AgentToolkitProjection.Tag
    return yield* projection.changesForFork(forkId).pipe(
      Stream.filterMap((toolkit) => {
        if (toolkit.config === null) return Option.none()
        const slot = getSlotStateForRole(toolkit.config, roleId)
        return slot._tag === 'Pending'
          ? Option.none()
          : Option.some({ slot, toolkit })
      }),
      Stream.runHead,
      Effect.flatMap(Option.match({
        onNone: () => Effect.dieMessage('Agent toolkit observation ended before model configuration settled'),
        onSome: Effect.succeed,
      })),
    )
  })
}
