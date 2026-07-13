import { Effect, Context } from 'effect'
import type { AgentRoutingState } from '../projections/agent-routing'
import type { AgentLifecycleState } from '../projections/agent-lifecycle'

export interface ProjectionReader {
  getAgentRouting(): Effect.Effect<AgentRoutingState>
  getAgentState(): Effect.Effect<AgentLifecycleState>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}