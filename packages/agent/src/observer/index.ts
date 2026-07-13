/**
 * Observer escalation system — public exports.
 */

// Justification types and templates
export type { ObserverJustification } from './justifications'
export { JUSTIFICATION_TEMPLATES, renderEscalationMessage, renderEscalationMessages } from './justifications'

// Event/message types
export type { ObserverTurnData, ObserverOutcome } from './types'

// Toolkit
export { observerToolkit, type PassInput, type EscalateInput } from './schema'

// Prompt and rendering
export { getObserverSystemPrompt, observerWindowToPrompt } from './prompt'

// State
export { ObserverStateTag, ObserverStateLive } from './state'

// Worker
export { ObserverWorker } from './worker'
