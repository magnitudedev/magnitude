import { Effect } from 'effect'
import { createObservable } from '@magnitudedev/agent-definition'
import { ProjectionReaderTag } from './projection-reader'
import { formatAgentsStatus } from '../prompts/agents'

export const agentsStatusObservable = createObservable({
  name: 'agents-status',
  observe: () => Effect.gen(function* () {
    const reader = yield* ProjectionReaderTag
    const registry = yield* reader.getAgentRegistry()
    const agents = Array.from(registry.agents.values())
    const formatted = formatAgentsStatus(agents)
    if (!formatted) return []
    return [{ type: 'text' as const, text: formatted }]
  })
})