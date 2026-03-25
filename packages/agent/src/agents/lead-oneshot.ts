import { defineRole } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import oneshotPromptRaw from './prompts/lead-oneshot.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import {
  constraintsLens,
  leadObservables,
  leadPolicy,
  leadTools,
  leadTurnPolicy,
  pivotLens,
  strategyLens,
  turnLens,
  validationLens,
} from './lead-shared'

const systemPrompt = compilePromptTemplate(oneshotPromptRaw)

export const leadOneshotRole = defineRole<typeof leadTools, 'lead', PolicyContext>({
  tools: leadTools,
  id: 'lead-oneshot',
  slot: 'lead',
  systemPrompt,
  lenses: [constraintsLens, pivotLens, strategyLens, validationLens, turnLens],
  defaultRecipient: 'user',
  protocolRole: 'oneshot-lead',
  initialContext: { parentConversation: false },
  spawnable: false,
  observables: leadObservables,
  policy: leadPolicy,
  turn: leadTurnPolicy,
})
