import type { AgentLifecycleState } from './projections/agent-lifecycle'
import type { CompactionState } from './projections/compaction'
import type { ForkTurnState } from './projections/turn'

export type SessionWorkStatus =
  | { readonly _tag: 'Working'; readonly workerCount: number }
  | { readonly _tag: 'Quiescent'; readonly workerCount: 0 }

export interface SessionWorkSnapshot {
  readonly turns: ReadonlyMap<string | null, ForkTurnState>
  readonly agents: AgentLifecycleState
  readonly compactions: ReadonlyMap<string | null, CompactionState>
  readonly detachedProcessCount: number
}

/**
 * The one authoritative derivation of whether a session owns continuing work.
 * Keep this semantic: callers must never substitute traffic or client presence.
 */
export function deriveSessionWorkStatus(snapshot: SessionWorkSnapshot): SessionWorkStatus {
  const workingAgents = [...snapshot.agents.agents.values()].filter((agent) => agent.status === 'working').length
  const turnWork = [...snapshot.turns.values()].some(
    (turn) => turn._tag === 'active' || turn._tag === 'interrupting' || turn.triggers.length > 0
  )
  const compactionWork = [...snapshot.compactions.values()].some(
    (compaction) => compaction._tag !== 'idle' || compaction.shouldCompact === true
  )
  const working = turnWork || workingAgents > 0 || compactionWork || snapshot.detachedProcessCount > 0

  return working
    ? {
        _tag: 'Working',
        workerCount: workingAgents,
      }
    : {
        _tag: 'Quiescent',
        workerCount: 0,
      }
}
