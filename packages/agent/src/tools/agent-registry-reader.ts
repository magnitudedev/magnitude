/**
 * Agent Registry State Reader Service
 *
 * Service tag for tools to access agent registry projection state.
 */

import { Effect, Context } from 'effect'
import type { AgentStatusState } from '../projections/agent-status'

export interface AgentRegistryStateReader {
  readonly getState: () => Effect.Effect<AgentStatusState>
}

export class AgentRegistryStateReaderTag extends Context.Tag('AgentRegistryStateReader')<
  AgentRegistryStateReaderTag,
  AgentRegistryStateReader
>() {}
