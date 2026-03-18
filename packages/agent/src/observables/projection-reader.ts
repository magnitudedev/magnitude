import { Effect, Context } from 'effect'
import type { AgentRoutingState } from '../projections/agent-routing'
import type { AgentStatusState } from '../projections/agent-status'
import type { BackgroundProcessState } from '../projections/background-processes'

export interface ProjectionReader {
  getAgentRouting(): Effect.Effect<AgentRoutingState>
  getAgentStatus(): Effect.Effect<AgentStatusState>
  getBackgroundProcesses(): Effect.Effect<Map<number, BackgroundProcessState>>
}

export class ProjectionReaderTag extends Context.Tag('ProjectionReader')<ProjectionReaderTag, ProjectionReader>() {}