/**
 * Lead Agent Definition
 *
 * The user-facing brain. Manages proposals, dispatches agents,
 * and communicates with user. Does not directly edit code — delegates
 * to sub-agents via the agent tools.
 */

import { defineRole } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import leadPromptRaw from './prompts/lead.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import {
  alignmentLens,
  tasksLens,
  diligenceLens,
  skillsLens,
  pivotLens,
  turnLens,
  leadObservables,
  leadPolicy,
  leadTools,
  leadTurnPolicy,
} from './lead-shared'

const systemPrompt = compilePromptTemplate(leadPromptRaw)

export const leadRole = defineRole<typeof leadTools, 'lead', PolicyContext>({
  tools: leadTools,
  id: 'lead',
  slot: 'lead',
  systemPrompt,
  lenses: [alignmentLens, tasksLens, diligenceLens, skillsLens, pivotLens, turnLens],
  defaultRecipient: 'user',
  protocolRole: 'lead',
  initialContext: { parentConversation: false },
  observables: leadObservables,
  policy: leadPolicy,
  turn: leadTurnPolicy,
})
