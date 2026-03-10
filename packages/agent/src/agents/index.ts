/**
 * Agent Registry
 *
 * All agent definitions, accessible by variant.
 * Loads prompts and creates definitions at registration time.
 */

import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'
import { actionsTagOpen, actionsTagClose, thinkTagOpen, thinkTagClose, commsTagOpen, commsTagClose } from '@magnitudedev/xml-act'
import type { PolicyContext } from './types'
import { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE } from '../constants'
import { createBrowser } from './browser'
import builderPromptRaw from './prompts/builder.txt'
import browserPromptRaw from './prompts/browser.txt'
import { createBuilder } from './builder'
import { createDebugger } from './debugger'
import debuggerPromptRaw from './prompts/debugger.txt'
import orchestratorPromptRaw from './prompts/orchestrator.txt'
import { createOrchestrator } from './orchestrator'
import { createPlanner } from './planner'
import plannerPromptRaw from './prompts/planner.txt'
import explorerPromptRaw from './prompts/explorer.txt'
import { createExplorer } from './explorer'
import reviewerPromptRaw from './prompts/reviewer.txt'
import { createReviewer } from './reviewer'

export type AgentVariant = 'orchestrator' | 'builder' | 'explorer' | 'planner' | 'debugger' | 'reviewer' | 'browser'

const replacePromptTokens = (raw: string): string =>
  raw
    .replaceAll('{{PROSE_OPEN}}', PROSE_DELIM_OPEN)
    .replaceAll('{{PROSE_CLOSE}}', PROSE_DELIM_CLOSE)
    .replaceAll('{{ACTIONS_OPEN}}', actionsTagOpen())
    .replaceAll('{{ACTIONS_CLOSE}}', actionsTagClose())
    .replaceAll('{{THINK_OPEN}}', thinkTagOpen())
    .replaceAll('{{THINK_CLOSE}}', thinkTagClose())
    .replaceAll('{{COMMS_OPEN}}', commsTagOpen())
    .replaceAll('{{COMMS_CLOSE}}', commsTagClose())

const ORCHESTRATOR_PROMPT = replacePromptTokens(orchestratorPromptRaw)
const BUILDER_PROMPT = replacePromptTokens(builderPromptRaw)
const EXPLORER_PROMPT = replacePromptTokens(explorerPromptRaw)
const PLANNER_PROMPT = replacePromptTokens(plannerPromptRaw)
const DEBUGGER_PROMPT = replacePromptTokens(debuggerPromptRaw)
const REVIEWER_PROMPT = replacePromptTokens(reviewerPromptRaw)
const BROWSER_PROMPT = replacePromptTokens(browserPromptRaw)

// =============================================================================
// Registry (lazy init)
// =============================================================================

/** All Magnitude agents use PolicyContext as framework-provided context. */
type MagnitudeAgentDef = AgentDefinition<ToolSet, PolicyContext>

let _agents: Record<AgentVariant, MagnitudeAgentDef> | null = null

function getAgents(): Record<AgentVariant, MagnitudeAgentDef> {
  if (!_agents) {
    _agents = {
      orchestrator: createOrchestrator(ORCHESTRATOR_PROMPT),
      builder: createBuilder(BUILDER_PROMPT),
      explorer: createExplorer(EXPLORER_PROMPT),
      planner: createPlanner(PLANNER_PROMPT),
      debugger: createDebugger(DEBUGGER_PROMPT),
      reviewer: createReviewer(REVIEWER_PROMPT),
      browser: createBrowser(BROWSER_PROMPT),
    }
  }
  return _agents
}

// =============================================================================
// Dynamic Overrides (for evals / testing)
// =============================================================================

const _overrides = new Map<string, MagnitudeAgentDef>()

/**
 * Register a custom agent definition that overrides a built-in variant.
 * Takes precedence over built-in definitions in getAgentDefinition().
 */
export function registerAgentDefinition(name: string, def: MagnitudeAgentDef): void {
  _overrides.set(name, def)
}

/** Clear all dynamic agent overrides. */
export function clearAgentOverrides(): void {
  _overrides.clear()
}

export function getAgentDefinition(variant: AgentVariant): MagnitudeAgentDef {
  const override = _overrides.get(variant)
  if (override) return override
  return getAgents()[variant]
}
