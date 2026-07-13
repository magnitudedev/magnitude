/**
 * Fork Services
 *
 * AgentStateReaderTag — service for tools to access agent projection state
 */

import { Effect, Context } from 'effect'
import type { AgentInfo, AgentLifecycleState } from '../projections/agent-lifecycle'

// =============================================================================
// Fork State Reader Service (for tools to access fork projection state)
// =============================================================================

export interface AgentStateReader {
  readonly getAgentState: () => Effect.Effect<AgentLifecycleState>
  readonly getAgent: (agentId: string) => Effect.Effect<AgentInfo | undefined>
}

export class AgentStateReaderTag extends Context.Tag('AgentStateReader')<
  AgentStateReaderTag,
  AgentStateReader
>() {}