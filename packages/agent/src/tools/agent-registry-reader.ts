/**
 * Agent Registry State Reader Service
 *
 * Service tag for tools to access agent registry projection state.
 */

import { Effect, Context } from 'effect'
import type { AgentState } from '../projections/agent'

export interface AgentRegistryStateReader {
  readonly getState: () => Effect.Effect<AgentState>
}

export class AgentRegistryStateReaderTag extends Context.Tag('AgentRegistryStateReader')<
  AgentRegistryStateReaderTag,
  AgentRegistryStateReader
>() {}
