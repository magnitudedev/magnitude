import { Effect, Context } from 'effect'
import type { AgentRoutingState } from '../projections/agent-routing'
import type { AgentStatusState } from '../projections/agent-status'

export interface ProjectionReader {
  getAgentRouting(): Effect.Effect<AgentRoutingState>
  getAgentStatus(): Effect.Effect<AgentStatusState>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}