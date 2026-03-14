/**
 * Debug Introspection Service
 *
 * Provides debug visibility into all projection states for development.
 * Only initialized when debug mode is enabled (--debug flag).
 */

import { Effect, Stream, SubscriptionRef } from 'effect'
import { DisplayProjection } from './display'
import { AgentRoutingProjection } from './agent-routing'

import { TurnProjection } from './turn'
import { MemoryProjection } from './memory'
import { CompactionProjection } from './compaction'
import { WorkingStateProjection } from './working-state'
import { SessionContextProjection } from './session-context'

import { ArtifactProjection } from './artifact'
import { ChatTitleProjection } from './chat-title'
import { ReplayProjection } from './replay'

import { CHARS_PER_TOKEN } from '../constants'
import { getContextLimits } from '../constants'

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

// =============================================================================
// Helpers
// =============================================================================

interface ResolvedProjections {
  displayProj: Effect.Effect.Success<typeof DisplayProjection.Tag>
  agentProj: Effect.Effect.Success<typeof AgentRoutingProjection.Tag>

  turnProj: Effect.Effect.Success<typeof TurnProjection.Tag>
  memoryProj: Effect.Effect.Success<typeof MemoryProjection.Tag>
  compactionProj: Effect.Effect.Success<typeof CompactionProjection.Tag>
  workingProj: Effect.Effect.Success<typeof WorkingStateProjection.Tag>
  sessionProj: Effect.Effect.Success<typeof SessionContextProjection.Tag>

  artifactProj: Effect.Effect.Success<typeof ArtifactProjection.Tag>
  chatTitleProj: Effect.Effect.Success<typeof ChatTitleProjection.Tag>
  replayProj: Effect.Effect.Success<typeof ReplayProjection.Tag>

}

function resolveProjections() {
  return Effect.gen(function* () {
    return {
      displayProj: yield* DisplayProjection.Tag,
      agentProj: yield* AgentRoutingProjection.Tag,

      turnProj: yield* TurnProjection.Tag,
      memoryProj: yield* MemoryProjection.Tag,
      compactionProj: yield* CompactionProjection.Tag,
      workingProj: yield* WorkingStateProjection.Tag,
      sessionProj: yield* SessionContextProjection.Tag,

      artifactProj: yield* ArtifactProjection.Tag,
      chatTitleProj: yield* ChatTitleProjection.Tag,
      replayProj: yield* ReplayProjection.Tag,

    } satisfies ResolvedProjections
  })
}

function buildSnapshot(
  forkId: string | null,
  projs: ResolvedProjections
): Effect.Effect<DebugSnapshot> {
  return Effect.gen(function* () {
    const timestamp = Date.now()

    // Use SubscriptionRef.get for all projections — works for both standard and forked
    const displayRaw = yield* SubscriptionRef.get(projs.displayProj.state)
    const agentState = yield* SubscriptionRef.get(projs.agentProj.state)

    const turnRaw = yield* SubscriptionRef.get(projs.turnProj.state)
    const memoryRaw = yield* SubscriptionRef.get(projs.memoryProj.state)
    const compactionRaw = yield* SubscriptionRef.get(projs.compactionProj.state)
    const workingRaw = yield* SubscriptionRef.get(projs.workingProj.state)
    const sessionState = yield* SubscriptionRef.get(projs.sessionProj.state)

    const artifactState = yield* SubscriptionRef.get(projs.artifactProj.state)
    const chatTitleState = yield* SubscriptionRef.get(projs.chatTitleProj.state)
    const replayRaw = yield* SubscriptionRef.get(projs.replayProj.state)


    // Extract fork-specific state from forked projections
    const displayForkState = displayRaw.forks.get(forkId)
    const turnForkState = turnRaw.forks.get(forkId)
    const memoryForkState = memoryRaw.forks.get(forkId)
    const compactionForkState = compactionRaw.forks.get(forkId)
    const workingForkState = workingRaw.forks.get(forkId)
    const replayForkState = replayRaw.forks.get(forkId)


    const projections: ProjectionSnapshot[] = [

      { name: 'WorkingStateProjection', state: workingForkState, timestamp },
      { name: 'AgentRoutingProjection', state: agentState, timestamp },

      { name: 'TurnProjection', state: turnForkState, timestamp },
      { name: 'MemoryProjection', state: memoryForkState, timestamp },
      { name: 'CompactionProjection', state: compactionForkState, timestamp },
      { name: 'DisplayProjection', state: displayForkState, timestamp },
      { name: 'ArtifactProjection', state: artifactState, timestamp },
      { name: 'SessionContextProjection', state: sessionState, timestamp },
      { name: 'ChatTitleProjection', state: chatTitleState, timestamp },
      { name: 'ReplayProjection', state: replayForkState, timestamp },

    ]

    let contextUsage: ContextUsage | undefined
    if (memoryForkState && compactionForkState) {
      const limits = getContextLimits()
      contextUsage = {
        currentTokens: compactionForkState.tokenEstimate,
        hardCap: limits.hardCap,
        softCap: limits.softCap,
        messageCount: memoryForkState.messages.length,
        usagePercent: Math.round((compactionForkState.tokenEstimate / limits.hardCap) * 100),
        shouldCompact: compactionForkState.shouldCompact,
        isCompacting: compactionForkState.isCompacting,
      }
    }

    return { projections, contextUsage, timestamp }
  })
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Create a debug introspection stream that aggregates all projection states.
 */
export function createDebugStream(forkId: string | null) {
  return Effect.gen(function* () {
    const projs = yield* resolveProjections()

    const toTrigger = <A>(s: Stream.Stream<A>): Stream.Stream<void> => Stream.map(s, () => undefined as void)

    const mergedStream = Stream.mergeAll([
      toTrigger(projs.displayProj.state.changes),
      toTrigger(projs.agentProj.state.changes),

      toTrigger(projs.turnProj.state.changes),
      toTrigger(projs.memoryProj.state.changes),
      toTrigger(projs.compactionProj.state.changes),
      toTrigger(projs.workingProj.state.changes),
      toTrigger(projs.sessionProj.state.changes),

      toTrigger(projs.artifactProj.state.changes),
      toTrigger(projs.chatTitleProj.state.changes),
      toTrigger(projs.replayProj.state.changes),

    ], { concurrency: 'unbounded' })

    const debouncedStream = Stream.debounce(mergedStream, '100 millis')

    return Stream.mapEffect(debouncedStream, () => buildSnapshot(forkId, projs))
  })
}

/**
 * Get a one-time snapshot of all projection states.
 */
export function getDebugSnapshot(forkId: string | null) {
  return Effect.gen(function* () {
    const projs = yield* resolveProjections()
    return yield* buildSnapshot(forkId, projs)
  })
}