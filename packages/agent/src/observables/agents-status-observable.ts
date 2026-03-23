import { Effect } from 'effect'
import { createObservable } from '@magnitudedev/roles'
import { ProjectionReaderTag } from './projection-reader'
import { formatAgentsStatus } from '../prompts/agents'

export const agentsStatusObservable = createObservable({
  name: 'agents-status',
  observe: () => Effect.gen(function* () {
    const reader = yield* ProjectionReaderTag
    const statuses = yield* reader.getAgentStatus()
    const agents = Array.from(statuses.agents.values()).map(agent => ({
      agentId: agent.agentId,
      type: agent.role,
      status: agent.status,
    }))
    const formatted = formatAgentsStatus(agents)
    if (!formatted) return []
    return [{ type: 'text' as const, text: formatted }]
  })
})