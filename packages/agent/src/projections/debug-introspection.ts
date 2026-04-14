/**
 * Debug Introspection Service
 *
 * Provides debug visibility into all projection states for development.
 * Only initialized when debug mode is enabled (--debug flag).
 */

import { Effect, Stream, SubscriptionRef } from 'effect'
import { AmbientServiceTag } from '@magnitudedev/event-core'
import { DisplayProjection } from './display'
import { AgentRoutingProjection } from './agent-routing'
import { AgentStatusProjection } from './agent-status'
import { TurnProjection } from './turn'
import { MemoryProjection } from './memory'
import { CompactionProjection } from './compaction'

import { SessionContextProjection } from './session-context'
import { ReplayProjection } from './replay'
import { ConfigAmbient, getSlotConfig } from '../ambient/config-ambient'
import { getForkInfo } from '../agents'

// =============================================================================
// Types
// =============================================================================

export interface ProjectionSnapshot {
  readonly name: string
  readonly state: unknown
  readonly timestamp: number
}

export interface ContextUsage {
  readonly currentTokens: number
  readonly hardCap: number
  readonly softCap: number
  readonly messageCount: number
  readonly usagePercent: number
  readonly shouldCompact: boolean
  readonly isCompacting: boolean
}

export interface DebugSnapshot {
  readonly projections: readonly ProjectionSnapshot[]
  readonly contextUsage?: ContextUsage
  readonly timestamp: number
}

interface ResolvedProjections {
  displayProj: Effect.Effect.Success<typeof DisplayProjection.Tag>
  routingProj: Effect.Effect.Success<typeof AgentRoutingProjection.Tag>
  statusProj: Effect.Effect.Success<typeof AgentStatusProjection.Tag>
  turnProj: Effect.Effect.Success<typeof TurnProjection.Tag>
  memoryProj: Effect.Effect.Success<typeof MemoryProjection.Tag>
  compactionProj: Effect.Effect.Success<typeof CompactionProjection.Tag>

  sessionProj: Effect.Effect.Success<typeof SessionContextProjection.Tag>
  replayProj: Effect.Effect.Success<typeof ReplayProjection.Tag>
}

function resolveProjections() {
  return Effect.gen(function* () {
    return {
      displayProj: yield* DisplayProjection.Tag,
      routingProj: yield* AgentRoutingProjection.Tag,
      statusProj: yield* AgentStatusProjection.Tag,
      turnProj: yield* TurnProjection.Tag,
      memoryProj: yield* MemoryProjection.Tag,
      compactionProj: yield* CompactionProjection.Tag,

      sessionProj: yield* SessionContextProjection.Tag,
      replayProj: yield* ReplayProjection.Tag,
    } satisfies ResolvedProjections
  })
}

function buildSnapshot(
  forkId: string | null,
  projs: ResolvedProjections
) {
  return Effect.gen(function* () {
    const timestamp = Date.now()

    const displayRaw = yield* SubscriptionRef.get(projs.displayProj.state)
    const routingState = yield* SubscriptionRef.get(projs.routingProj.state)
    const statusState = yield* SubscriptionRef.get(projs.statusProj.state)
    const turnRaw = yield* SubscriptionRef.get(projs.turnProj.state)
    const memoryRaw = yield* SubscriptionRef.get(projs.memoryProj.state)
    const compactionRaw = yield* SubscriptionRef.get(projs.compactionProj.state)

    const sessionState = yield* SubscriptionRef.get(projs.sessionProj.state)
    const replayRaw = yield* SubscriptionRef.get(projs.replayProj.state)

    const displayForkState = displayRaw.forks.get(forkId)
    const turnForkState = turnRaw.forks.get(forkId)
    const memoryForkState = memoryRaw.forks.get(forkId)
    const compactionForkState = compactionRaw.forks.get(forkId)

    const replayForkState = replayRaw.forks.get(forkId)

    const projections: ProjectionSnapshot[] = [

      { name: 'AgentRoutingProjection', state: routingState, timestamp },
      { name: 'AgentStatusProjection', state: statusState, timestamp },
      { name: 'TurnProjection', state: turnForkState, timestamp },
      { name: 'MemoryProjection', state: memoryForkState, timestamp },
      { name: 'CompactionProjection', state: compactionForkState, timestamp },
      { name: 'DisplayProjection', state: displayForkState, timestamp },
      { name: 'SessionContextProjection', state: sessionState, timestamp },
      { name: 'ReplayProjection', state: replayForkState, timestamp },
    ]

    let contextUsage: ContextUsage | undefined
    if (memoryForkState && compactionForkState) {
      const ambientService = yield* AmbientServiceTag
      const configState = ambientService.getValue(ConfigAmbient)
      const info = getForkInfo(statusState, forkId)
      const limits = info ? getSlotConfig(configState, info.slot) : getSlotConfig(configState, 'lead')
      contextUsage = {
        currentTokens: compactionForkState.tokenEstimate,
        hardCap: limits.hardCap,
        softCap: limits.softCap,
        messageCount: memoryForkState.messages.length,
        usagePercent: Math.round((compactionForkState.tokenEstimate / limits.hardCap) * 100),
        shouldCompact: compactionForkState.shouldCompact,
        isCompacting: compactionForkState._tag !== 'idle',
      }
    }

    return { projections, contextUsage, timestamp }
  })
}

export function createDebugStream(forkId: string | null) {
  return Effect.gen(function* () {
    const projs = yield* resolveProjections()
    const toTrigger = <A>(s: Stream.Stream<A>): Stream.Stream<void> => Stream.map(s, () => undefined as void)

    const mergedStream = Stream.mergeAll([
      toTrigger(projs.displayProj.state.changes),
      toTrigger(projs.routingProj.state.changes),
      toTrigger(projs.statusProj.state.changes),
      toTrigger(projs.turnProj.state.changes),
      toTrigger(projs.memoryProj.state.changes),
      toTrigger(projs.compactionProj.state.changes),

      toTrigger(projs.sessionProj.state.changes),
      toTrigger(projs.replayProj.state.changes),
    ], { concurrency: 'unbounded' })

    const debouncedStream = Stream.debounce(mergedStream, '100 millis')
    return Stream.mapEffect(debouncedStream, () => buildSnapshot(forkId, projs))
  })
}

export function getDebugSnapshot(forkId: string | null) {
  return Effect.gen(function* () {
    const projs = yield* resolveProjections()
    return yield* buildSnapshot(forkId, projs)
  })
}