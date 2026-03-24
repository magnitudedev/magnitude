import { defineRole } from '@magnitudedev/roles'
import type { PolicyContext } from './types'
import oneshotPromptRaw from './prompts/orchestrator-oneshot.txt' with { type: 'text' }
import { compilePromptTemplate } from '../prompts/system-prompt'
import {
  constraintsLens,
  orchestratorObservables,
  orchestratorPermission,
  orchestratorTools,
  orchestratorTurnPolicy,
  pivotLens,
  strategyLens,
  turnLens,
  validationLens,
} from './orchestrator-shared'

const systemPrompt = compilePromptTemplate(oneshotPromptRaw)

export const orchestratorOneshotRole = defineRole<typeof orchestratorTools, 'orchestrator', PolicyContext>({
  tools: orchestratorTools,
  id: 'orchestrator-oneshot',
  slot: 'orchestrator',
  systemPrompt,
  lenses: [constraintsLens, pivotLens, strategyLens, validationLens, turnLens],
  defaultRecipient: 'user',
  protocolRole: 'oneshot-orchestrator',
  initialContext: { parentConversation: false },
  spawnable: false,
  observables: orchestratorObservables,
  permission: orchestratorPermission,
  turn: orchestratorTurnPolicy,
})
