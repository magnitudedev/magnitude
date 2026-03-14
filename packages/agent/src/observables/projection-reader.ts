import { Effect, Context } from 'effect'
import type { AgentRoutingState } from '../projections/agent-routing'

export interface ProjectionReader {
  getAgentRegistry(): Effect.Effect<AgentRoutingState>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}