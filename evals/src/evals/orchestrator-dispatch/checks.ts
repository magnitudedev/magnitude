/**
 * Orchestrator Dispatch Check Factories
 *
 * Each factory returns a Check that parses the raw orchestrator XML response
 * and asserts a specific behavioral property.
 */

import type { Check } from '../../types'
import { parseOrchestratorResponse } from './xml-parser'

// =============================================================================
// Structural
// =============================================================================

export function hasThinkBlock(): Check {
  return {
    id: 'has-think-block',
    description: 'Response contains a <think> block',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      return {
        passed: parsed.hasThinkBlock,
        message: parsed.hasThinkBlock ? undefined : 'No <think> block found',
      }
    },
  }
}

export function hasUserMessage(): Check {
  return {
    id: 'has-user-message',
    description: 'Response includes a user-facing message',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      return {
        passed: parsed.hasUserMessage,
        message: parsed.hasUserMessage ? undefined : 'No user-facing message found',
      }
    },
  }
}

// =============================================================================
// Agent deployment
// =============================================================================

export function noAgentsDeployed(): Check {
  return {
    id: 'no-agents-deployed',
    description: 'No subagents were deployed',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const count = parsed.agentCreates.length
      return {
        passed: count === 0,
        message: count === 0 ? undefined : `Deployed ${count} agent(s): ${parsed.agentTypesInOrder.join(', ')}`,
      }
    },
  }
}

export function agentTypeDeployed(type: string): Check {
  return {
    id: `deploys-${type}`,
    description: `Deploys a ${type} agent`,
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const found = parsed.agentCreates.some(a => a.type === type)
      return {
        passed: found,
        message: found ? undefined : `No ${type} agent deployed. Agents: ${parsed.agentTypesInOrder.join(', ') || 'none'}`,
      }
    },
  }
}

export function agentTypeNotDeployed(type: string): Check {
  return {
    id: `no-${type}`,
    description: `Does NOT deploy a ${type} agent`,
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const found = parsed.agentCreates.some(a => a.type === type)
      return {
        passed: !found,
        message: !found ? undefined : `${type} agent was deployed but should not have been`,
      }
    },
  }
}

export function agentOrderedBefore(typeA: string, typeB: string): Check {
  return {
    id: `${typeA}-before-${typeB}`,
    description: `${typeA} agent is deployed before ${typeB} agent`,
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const posA = parsed.agentCreates.find(a => a.type === typeA)?.position
      const posB = parsed.agentCreates.find(a => a.type === typeB)?.position
      if (posA === undefined) return { passed: false, message: `No ${typeA} agent found` }
      if (posB === undefined) return { passed: false, message: `No ${typeB} agent found` }
      return {
        passed: posA < posB,
        message: posA < posB ? undefined : `${typeA} (pos ${posA}) appears after ${typeB} (pos ${posB})`,
      }
    },
  }
}

// =============================================================================
// Proposal
// =============================================================================

export function proposeCalled(): Check {
  return {
    id: 'propose-called',
    description: 'Calls <propose> to submit a proposal',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const found = parsed.proposes.length > 0
      return {
        passed: found,
        message: found ? undefined : 'No <propose> call found',
      }
    },
  }
}

export function proposeNotCalled(): Check {
  return {
    id: 'no-propose',
    description: 'Does NOT call <propose>',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const found = parsed.proposes.length > 0
      return {
        passed: !found,
        message: !found ? undefined : '<propose> was called but should not have been',
      }
    },
  }
}

export function proposeHasCriteria(): Check {
  return {
    id: 'propose-has-criteria',
    description: 'Proposal includes acceptance criteria',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const propose = parsed.proposes[0]
      if (!propose) return { passed: false, message: 'No <propose> call found' }
      return {
        passed: propose.hasCriteria,
        message: propose.hasCriteria ? undefined : 'Proposal has no <criterion> children',
      }
    },
  }
}

export function proposeHasArtifact(): Check {
  return {
    id: 'propose-has-artifact',
    description: 'Proposal includes attached artifacts',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const propose = parsed.proposes[0]
      if (!propose) return { passed: false, message: 'No <propose> call found' }
      return {
        passed: propose.hasArtifacts,
        message: propose.hasArtifacts ? undefined : 'Proposal has no <artifact> children',
      }
    },
  }
}

// =============================================================================
// Direct tool usage
// =============================================================================

export function usedDirectTools(): Check {
  return {
    id: 'used-direct-tools',
    description: 'Orchestrator uses its own tools directly (fs-read, shell, etc.)',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const found = parsed.directToolUses.length > 0
      return {
        passed: found,
        message: found ? `Used: ${parsed.directToolUses.join(', ')}` : 'No direct tool usage found',
      }
    },
  }
}

export function noDirectFileEdits(): Check {
  return {
    id: 'no-direct-file-edits',
    description: 'Orchestrator does NOT edit files directly',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const editTools = parsed.directToolUses.filter(t => t === 'fs-write' || t === 'fs-edit')
      return {
        passed: editTools.length === 0,
        message: editTools.length === 0 ? undefined : `Orchestrator directly edited files: ${editTools.join(', ')}`,
      }
    },
  }
}
