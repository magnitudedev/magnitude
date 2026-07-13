import { AmbientServiceTag, Introspection, type Projection } from '@magnitudedev/event-core'
import { Data, Effect, Option, Stream } from 'effect'
import { ConfigAmbient, getSlotConfigForRole } from '../ambient/config-ambient'
import { getForkInfo } from '../agents/registry'
import { DisplayTimelineProjection } from '../display'
import type { DisplayTimelineState } from '../display'
import { CompactionProjection, type CompactionState } from '../projections/compaction'
import { AgentLifecycleProjection, type AgentLifecycleState } from '../projections/agent-lifecycle'
import { WindowProjection } from '../window'
import type { ForkWindowState } from '../window'
import {
  currentAddressedSpaceIntrospection,
} from './addressed'
import {
  createAddressedAtlas,
  type AddressedAtlasNode,
} from './addressed-tree'

export interface ProjectionSummary {
  readonly kind: 'null' | 'undefined' | 'string' | 'number' | 'boolean' | 'array' | 'map' | 'object'
  readonly size: number | null
  readonly estimatedBytes: number | null
  readonly label: string
}

export interface ProjectionIntrospection extends Introspection.ProjectionIntrospection {
  readonly summary: ProjectionSummary
}

export interface RuntimeIntrospection {
  readonly engineName: string
  readonly schemaVersion: string
  readonly timestamp: number
  readonly projections: readonly ProjectionIntrospection[]
}

export interface ContextIntrospection {
  readonly currentTokens: number
  readonly hardCap: number
  readonly softCap: number
  readonly messageCount: number
  readonly usagePercent: number
  readonly shouldCompact: boolean
  readonly isCompacting: boolean
}

export interface DisplayIntrospection {
  readonly timelines: readonly {
    readonly forkId: string | null
    readonly mode: DisplayTimelineState['mode']
    readonly messageCount: number
    readonly streamingMessageId: string | null
  }[]
}

export interface AgentIntrospection {
  readonly timestamp: number
  readonly runtime: RuntimeIntrospection
  readonly projections: readonly ProjectionIntrospection[]
  readonly contextUsage?: ContextIntrospection
  readonly addressedAtlas: readonly AddressedAtlasNode[]
  readonly display: DisplayIntrospection
}

export class AgentIntrospectionError extends Data.TaggedError('AgentIntrospectionError')<{
  readonly operation: 'current' | 'changes'
  readonly cause: unknown
}> {}

const introspectionError = (operation: AgentIntrospectionError['operation']) =>
  (cause: unknown) => new AgentIntrospectionError({ operation, cause })

const projectionSummary = (state: unknown): ProjectionSummary => {
  const estimatedBytes = estimateJsonBytes(state)
  if (state === null) return { kind: 'null', size: null, estimatedBytes, label: 'empty' }
  if (state === undefined) return { kind: 'undefined', size: null, estimatedBytes, label: 'missing' }
  if (typeof state === 'string') return { kind: 'string', size: state.length, estimatedBytes, label: `${state.length} chars` }
  if (typeof state === 'number') return { kind: 'number', size: null, estimatedBytes, label: 'number' }
  if (typeof state === 'boolean') return { kind: 'boolean', size: null, estimatedBytes, label: 'boolean' }
  if (Array.isArray(state)) return { kind: 'array', size: state.length, estimatedBytes, label: `${state.length} items` }
  if (state instanceof Map) return { kind: 'map', size: state.size, estimatedBytes, label: `${state.size} entries` }
  if (typeof state === 'object') {
    const size = Object.keys(state as Record<string, unknown>).length
    return { kind: 'object', size, estimatedBytes, label: `${size} keys` }
  }
  return { kind: 'undefined', size: null, estimatedBytes, label: typeof state }
}

const estimateJsonBytes = (value: unknown): number | null => {
  try {
    const encoded = JSON.stringify(value)
    return encoded === undefined
      ? null
      : new TextEncoder().encode(encoded).byteLength
  } catch {
    return null
  }
}

const buildRuntimeIntrospection = (
  runtime: Introspection.RuntimeIntrospection
): RuntimeIntrospection => ({
  engineName: runtime.engineName,
  schemaVersion: runtime.schemaVersion,
  timestamp: runtime.timestamp,
  projections: runtime.projections.map((projection) => ({
    ...projection,
    summary: projectionSummary(projection.state),
  })),
})

const emptyRuntimeIntrospection = (): RuntimeIntrospection => ({
  engineName: 'unknown',
  schemaVersion: 'unknown',
  timestamp: Date.now(),
  projections: [],
})

const buildContextUsage = (
  forkId: string | null,
  statusState: AgentLifecycleState,
  memoryForkState: ForkWindowState,
  compactionForkState: CompactionState,
) =>
  Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    const configState = ambientService.getValue(ConfigAmbient)
    const info = getForkInfo(statusState, forkId)
    const limits = info ? getSlotConfigForRole(configState, info.roleId) : getSlotConfigForRole(configState, 'leader')
    return {
      currentTokens: memoryForkState.tokenEstimate,
      hardCap: limits.hardCap,
      softCap: limits.softCap,
      messageCount: memoryForkState.messages.length,
      usagePercent: Math.round((memoryForkState.tokenEstimate / limits.hardCap) * 100),
      shouldCompact: compactionForkState.shouldCompact,
      isCompacting: compactionForkState._tag !== 'idle',
    }
  })

const displayIntrospection = (
  forks: ReadonlyMap<string | null, DisplayTimelineState>
): DisplayIntrospection => ({
  timelines: [...forks].map(([forkId, fork]) => ({
    forkId,
    mode: fork.mode,
    messageCount: fork.messages.totalCount,
    streamingMessageId: fork.streamingMessageId,
  })),
})

interface AgentIntrospectionServices {
  readonly runtimeOption: Option.Option<Effect.Effect.Success<typeof Introspection.RuntimeIntrospector>>
  readonly status: {
    readonly get: Effect.Effect<AgentLifecycleState>
  }
  readonly window: {
    readonly getFork: (forkId: string | null) => Effect.Effect<ForkWindowState>
  }
  readonly compaction: {
    readonly getFork: (forkId: string | null) => Effect.Effect<CompactionState>
  }
  readonly displayTimeline: {
    readonly getAllForks: () => Effect.Effect<Map<string | null, DisplayTimelineState>>
    readonly state: {
      readonly changes: Stream.Stream<Projection.ForkedState<DisplayTimelineState>>
    }
  }
}

const readAgentIntrospection = (
  forkId: string | null,
  services: AgentIntrospectionServices,
) =>
  Effect.gen(function* () {
    const runtime = Option.isSome(services.runtimeOption)
      ? buildRuntimeIntrospection(yield* services.runtimeOption.value.current(forkId))
      : emptyRuntimeIntrospection()

    const statusState = yield* services.status.get
    const memoryForkState = yield* services.window.getFork(forkId)
    const compactionForkState = yield* services.compaction.getFork(forkId)
    const timelineForks = yield* services.displayTimeline.getAllForks()
    const addressedSpaces = yield* currentAddressedSpaceIntrospection
    const contextUsage = yield* buildContextUsage(forkId, statusState, memoryForkState, compactionForkState)
    const addressedAtlas = yield* createAddressedAtlas(timelineForks, addressedSpaces)

    return {
      timestamp: Date.now(),
      runtime,
      projections: runtime.projections,
      contextUsage,
      addressedAtlas,
      display: displayIntrospection(timelineForks),
    } satisfies AgentIntrospection
  })

export const getAgentIntrospection = (forkId: string | null) =>
  Effect.gen(function* () {
    const runtimeOption = yield* Effect.serviceOption(Introspection.RuntimeIntrospector)
    const status = yield* AgentLifecycleProjection.Tag
    const window = yield* WindowProjection.Tag
    const compaction = yield* CompactionProjection.Tag
    const displayTimeline = yield* DisplayTimelineProjection.Tag

    return yield* readAgentIntrospection(forkId, {
      runtimeOption,
      status,
      window,
      compaction,
      displayTimeline,
    })
  }).pipe(Effect.mapError(introspectionError('current')))

export const createAgentIntrospectionChanges = (forkId: string | null) =>
  Effect.gen(function* () {
    const runtimeOption = yield* Effect.serviceOption(Introspection.RuntimeIntrospector)
    const addressedOption = yield* Effect.serviceOption(Introspection.AddressedIntrospectionRegistry)
    const ambientService = yield* AmbientServiceTag
    const status = yield* AgentLifecycleProjection.Tag
    const window = yield* WindowProjection.Tag
    const compaction = yield* CompactionProjection.Tag
    const displayTimeline = yield* DisplayTimelineProjection.Tag
    const runtimeStream = Option.isSome(runtimeOption)
      ? runtimeOption.value.changes(forkId).pipe(Stream.map(() => undefined))
      : Stream.never
    const displayAddressStream = displayTimeline.state.changes.pipe(Stream.map(() => undefined))
    const triggers = Stream.merge(runtimeStream, displayAddressStream).pipe(
      Stream.debounce('100 millis')
    )

    let current = readAgentIntrospection(forkId, {
      runtimeOption,
      status,
      window,
      compaction,
      displayTimeline,
    }).pipe(
      Effect.mapError(introspectionError('current')),
      Effect.provideService(AmbientServiceTag, ambientService),
    )

    if (Option.isSome(addressedOption)) {
      current = current.pipe(
        Effect.provideService(
          Introspection.AddressedIntrospectionRegistry,
          addressedOption.value,
        ),
      )
    }

    return Stream.mapEffect(triggers, () => current)
  }).pipe(Effect.mapError(introspectionError('changes')))

export type {
  AddressedAtlasGroup,
  AddressedAtlasMetrics,
  AddressedAtlasNode,
  AddressedAtlasResident,
  AddressedAtlasSegment,
  AddressedPin,
} from './addressed-tree'
