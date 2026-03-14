/**
 * Fork Services
 *
 * AgentStateReaderTag — service for tools to access agent projection state
 */

import { Effect, Context } from 'effect'
import type { AgentStatusState } from '../projections/agent-status'

// =============================================================================
// Fork State Reader Service (for tools to access fork projection state)
// =============================================================================

export interface AgentStateReader {
  readonly getAgentState: () => Effect.Effect<AgentStatusState>
}

export class AgentStateReaderTag extends Context.Tag('AgentStateReader')<
  AgentStateReaderTag,
  AgentStateReader
>() {}