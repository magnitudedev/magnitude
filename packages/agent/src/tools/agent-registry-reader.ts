/**
 * Agent Registry State Reader Service
 *
 * Service tag for tools to access agent registry projection state.
 */

import { Effect, Context } from 'effect'
import type { AgentRegistryState } from '../projections/agent-registry'

export interface AgentRegistryStateReader {
  readonly getState: () => Effect.Effect<AgentRegistryState>
}

export class AgentRegistryStateReaderTag extends Context.Tag('AgentRegistryStateReader')<
  AgentRegistryStateReaderTag,
  AgentRegistryStateReader
>() {}
