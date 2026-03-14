/**
 * Agent Registry State Reader Service
 *
 * Service tag for tools to access agent registry projection state.
 */

import { Effect, Context } from 'effect'
import type { AgentRoutingState } from '../projections/agent-routing'

export interface AgentRegistryStateReader {
  readonly getState: () => Effect.Effect<AgentRoutingState>
}

export class AgentRegistryStateReaderTag extends Context.Tag('AgentRegistryStateReader')<
  AgentRegistryStateReaderTag,
  AgentRegistryStateReader
>() {}
