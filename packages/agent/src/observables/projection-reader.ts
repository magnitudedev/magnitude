import { Effect, Context } from 'effect'
import type { AgentState } from '../projections/agent'

export interface ProjectionReader {
  getAgentRegistry(): Effect.Effect<AgentState>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}