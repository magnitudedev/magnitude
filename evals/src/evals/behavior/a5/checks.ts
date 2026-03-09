import type { Check } from '../../types'
import { parseOrchestratorResponse } from './parser'

export function hasThinkBlock(): Check {
  return {
    id: 'has-think-block',
    description: 'Response contains a closed <think> block',
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      return {
        passed: parsed.hasThinkBlock,
        message: parsed.hasThinkBlock ? undefined : 'Missing closed <think> block',
      }
    },
  }
}

export function messagedUser(): Check {
  return {
    id: 'messaged-user',
    description: 'Response includes at least one <message to="user">',
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      return {
        passed: parsed.userMessages.length > 0,
        message: parsed.userMessages.length > 0 ? undefined : 'No user-directed message found',
      }
    },
  }
}

export function noAgentsDeployed(): Check {
  return {
    id: 'no-agents-deployed',
    description: 'No agents were deployed',
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      return {
        passed: parsed.agentCreates.length === 0,
        message: parsed.agentCreates.length === 0 ? undefined : `Deployed ${parsed.agentCreates.length} agent(s)`,
      }
    },
  }
}

export function agentDeployed(type: string): Check {
  return {
    id: `agent-deployed-${type}`,
    description: `Deploys a ${type} agent`,
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      const found = parsed.agentCreates.some(a => a.type === type)
      return {
        passed: found,
        message: found ? undefined : `No ${type} agent deployed`,
      }
    },
  }
}

export function agentNotDeployed(type: string): Check {
  return {
    id: `agent-not-deployed-${type}`,
    description: `Does not deploy a ${type} agent`,
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      const found = parsed.agentCreates.some(a => a.type === type)
      return {
        passed: !found,
        message: !found ? undefined : `${type} agent was deployed`,
      }
    },
  }
}

export function proposalMade(): Check {
  return {
    id: 'proposal-made',
    description: 'At least one proposal is made',
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      return {
        passed: parsed.proposes.length > 0,
        message: parsed.proposes.length > 0 ? undefined : 'No <propose> found',
      }
    },
  }
}

export function noProposal(): Check {
  return {
    id: 'no-proposal',
    description: 'No proposal is made',
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      return {
        passed: parsed.proposes.length === 0,
        message: parsed.proposes.length === 0 ? undefined : '<propose> was called',
      }
    },
  }
}

export function agentCountAtMost(n: number): Check {
  return {
    id: `agent-count-at-most-${n}`,
    description: `Deploys at most ${n} agent(s)`,
    evaluate(rawResponse) {
      const parsed = parseOrchestratorResponse(rawResponse)
      const count = parsed.agentCreates.length
      return {
        passed: count <= n,
        message: count <= n ? undefined : `Deployed ${count} agent(s), expected at most ${n}`,
      }
    },
  }
}