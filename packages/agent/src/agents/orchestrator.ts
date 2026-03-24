/**
 * Orchestrator Agent Definition
 *
 * The user-facing brain. Manages proposals, dispatches agents,
 * and communicates with user. Does not directly edit code — delegates
 * to sub-agents via the agent tools.
 */

import { defineRole } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import orchestratorPromptRaw from './prompts/orchestrator.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import {
  ideateLens,
  intentLens,
  orchestratorObservables,
  orchestratorPermission,
  orchestratorTools,
  orchestratorTurnPolicy,
  practicesLens,
  protocolLens,
  strategyLens,
  turnLens,
} from './orchestrator-shared'

const systemPrompt = compilePromptTemplate(orchestratorPromptRaw)

export const orchestratorRole = defineRole<typeof orchestratorTools, 'orchestrator', PolicyContext>({
  tools: orchestratorTools,
  id: 'orchestrator',
  slot: 'orchestrator',
  systemPrompt,
  lenses: [intentLens, ideateLens, strategyLens, protocolLens, practicesLens, turnLens],
  defaultRecipient: 'user',
  protocolRole: 'orchestrator',
  initialContext: { parentConversation: false },
  observables: orchestratorObservables,
  permission: orchestratorPermission,
  turn: orchestratorTurnPolicy,
})
