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

export function hasLensesBlock(): Check {
  return {
    id: 'has-lenses-block',
    description: 'Response contains a <lenses> thinking block',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      return {
        passed: parsed.hasThinkBlock,
        message: parsed.hasThinkBlock ? undefined : 'No <lenses> block found',
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

export function hasMessageToUser(): Check {
  return {
    id: 'has-message-to-user',
    description: 'Response includes <message to="user">...</message>',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const hasMessage = parsed.messages.some(msg => msg.to.toLowerCase() === 'user')
      return {
        passed: hasMessage,
        message: hasMessage ? undefined : 'No <message to="user"> found',
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

export function agentTypeDeployedCount(type: string, min = 1, max?: number): Check {
  return {
    id: `deploys-${type}-count-${min}-${max ?? 'inf'}`,
    description: `Deploys ${type} between ${min} and ${max ?? '∞'} times`,
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const count = parsed.agentCreates.filter(a => a.type === type).length
      const atLeast = count >= min
      const atMost = max === undefined || count <= max
      const passed = atLeast && atMost
      return {
        passed,
        message: passed ? undefined : `${type} deployed ${count} time(s), expected ${min}-${max ?? '∞'}`,
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

export function agentTypeNotDeployedBefore(typeForbidden: string, typeGate: string): Check {
  return {
    id: `${typeForbidden}-not-before-${typeGate}`,
    description: `${typeForbidden} is not deployed before ${typeGate}`,
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const forbidden = parsed.agentCreates.filter(a => a.type === typeForbidden).sort((a, b) => a.position - b.position)
      const gate = parsed.agentCreates.filter(a => a.type === typeGate).sort((a, b) => a.position - b.position)
      if (forbidden.length === 0) return { passed: true }
      if (gate.length === 0) return { passed: false, message: `No ${typeGate} agent found; ${typeForbidden} was deployed` }
      return {
        passed: forbidden[0].position > gate[0].position,
        message: forbidden[0].position > gate[0].position ? undefined : `${typeForbidden} appears before ${typeGate}`,
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

export function reviewerDeployedAfterBuilder(): Check {
  return {
    id: 'reviewer-after-builder',
    description: 'Reviewer is deployed after builder',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const firstBuilder = parsed.agentCreates.find(a => a.type === 'builder')
      const firstReviewer = parsed.agentCreates.find(a => a.type === 'reviewer')
      if (!firstBuilder) return { passed: false, message: 'No builder agent found' }
      if (!firstReviewer) return { passed: false, message: 'No reviewer agent found' }
      return {
        passed: firstReviewer.position > firstBuilder.position,
        message: firstReviewer.position > firstBuilder.position ? undefined : 'Reviewer deployed before builder',
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
      const editTools = parsed.directToolUses.filter(t => t === 'fs-write' || t === 'fs-edit' || t === 'write' || t === 'edit')
      return {
        passed: editTools.length === 0,
        message: editTools.length === 0 ? undefined : `Orchestrator directly edited files: ${editTools.join(', ')}`,
      }
    },
  }
}

export function hasNoDirectMutationToolsBeforeApproval(): Check {
  return {
    id: 'no-direct-mutations-before-approval',
    description: 'Avoid direct file mutation tools before explicit approval-like language',
    evaluate(raw) {
      const lower = raw.toLowerCase()
      const hasApprovalSignal = /\b(approve|approval|go ahead|go-ahead|sure,?\s*do it|yes,?\s*do it|looks good,?\s*proceed|proceed)\b/.test(lower)
      const parsed = parseOrchestratorResponse(raw)
      const hasMutations = parsed.fsEdits.length > 0 || parsed.fsWrites.length > 0
      if (hasApprovalSignal) return { passed: true }
      return {
        passed: !hasMutations,
        message: !hasMutations ? undefined : 'Detected direct mutation tools without approval-like signal',
      }
    },
  }
}

export function usesReadOnlyInvestigationBeforeExecution(): Check {
  return {
    id: 'read-only-investigation-before-execution',
    description: 'First action is read-only investigation, not execution',
    evaluate(raw) {
      const parsed = parseOrchestratorResponse(raw)
      const first = parsed.firstActionKind
      if (!first) return { passed: true }

      const readOnlyAction =
        first === 'tool:fs-read' ||
        first === 'tool:fs-search' ||
        first === 'tool:fs-tree' ||
        first === 'tool:shell' ||
        first === 'tool:artifact-read'

      if (readOnlyAction) return { passed: true }

      if (first === 'agent:create') {
        const firstAgent = parsed.agentCreates.sort((a, b) => a.position - b.position)[0]
        const type = firstAgent?.type
        const readOnlyAgent = type === 'explorer' || type === 'planner' || type === 'debugger' || type === 'researcher'
        return {
          passed: !!readOnlyAgent,
          message: readOnlyAgent ? undefined : `First deployed agent was ${type ?? 'unknown'}, expected investigation-oriented agent`,
        }
      }

      return {
        passed: false,
        message: `First action was ${first}, expected read-only investigation`,
      }
    },
  }
}