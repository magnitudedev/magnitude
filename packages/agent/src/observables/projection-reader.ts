import { Effect, Context } from 'effect'
import type { AgentRegistryState } from '../projections/agent-registry'

export interface ProjectionReader {
  getAgentRegistry(): Effect.Effect<AgentRegistryState>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}