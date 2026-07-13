/**
 * Observer escalation system — justifications and templates.
 *
 * The observer raises a single justification string when escalating.
 * The system templates the leader message from it — no observer-written prose.
 */

// =============================================================================
// Justification type
// =============================================================================

export type ObserverJustification =
  | 'difficulty'
  | 'churn'
  | 'frustration'

// =============================================================================
// Templates — what the leader sees for each justification
// =============================================================================

export const JUSTIFICATION_TEMPLATES: Record<ObserverJustification, string> = {
  difficulty:
    'System has detected a high-difficulty task that requires deeper reasoning than may be available in your current thread. The advisor operates at higher intelligence and can handle complex architectural decisions and difficult problems more effectively.',
  churn:
    'System has detected a high level of churn — repeated failed attempts, tunnel vision, or lack of a coherent strategy. The current approach may be fundamentally flawed. The advisor can provide a fresh perspective and help you step back to reassess.',
  frustration:
    'System has detected user frustration — the collaboration is not meeting expectations, you may be overstepping boundaries, or the user has corrected you repeatedly. The advisor can help you understand what went wrong and realign with the user\'s intent.',
}

// =============================================================================
// Rendering
// =============================================================================

/** Render a single justification into a <escalation_required> block. */
export function renderEscalationMessage(justification: ObserverJustification): string {
  return `<escalation_required>\n${JUSTIFICATION_TEMPLATES[justification]}\n</escalation_required>`
}

/** Render multiple justifications into a single <escalation_required> block. */
export function renderEscalationMessages(justifications: readonly ObserverJustification[]): string {
  if (justifications.length === 0) return ''
  if (justifications.length === 1) return renderEscalationMessage(justifications[0])
  const messages = justifications.map((j) => JUSTIFICATION_TEMPLATES[j]).join(' ')
  return `<escalation_required>\n${messages}\n</escalation_required>`
}
