/**
 * Fork Services
 *
 * AgentStateReaderTag — service for tools to access agent projection state
 */

import { Effect, Context } from 'effect'
import type { AgentInfo, AgentStatusState } from '../projections/agent-status'

// =============================================================================
// Fork State Reader Service (for tools to access fork projection state)
// =============================================================================

export interface AgentStateReader {
  readonly getAgentState: () => Effect.Effect<AgentStatusState>
  readonly getAgent: (agentId: string) => Effect.Effect<AgentInfo | undefined>
}

export class AgentStateReaderTag extends Context.Tag('AgentStateReader')<
  AgentStateReaderTag,
  AgentStateReader
>() {}