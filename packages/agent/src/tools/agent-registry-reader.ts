/**
 * Agent Registry State Reader Service
 *
 * Service tag for tools to access agent registry projection state.
 */

import { Effect, Context } from 'effect'
import type { AgentLifecycleState } from '../projections/agent-lifecycle'

export interface AgentRegistryStateReader {
  readonly getState: () => Effect.Effect<AgentLifecycleState>
}

export class AgentRegistryStateReaderTag extends Context.Tag('AgentRegistryStateReader')<
  AgentRegistryStateReaderTag,
  AgentRegistryStateReader
>() {}
