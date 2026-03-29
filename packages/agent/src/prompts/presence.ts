/**
 * User presence prompt formatting.
 *
 * Formats window focus state changes as system notifications for the orchestrator.
 */

export function formatUserPresence(focused: boolean): string {
  return focused
    ? 'The terminal window has regained focus. The user is likely present.'
    : 'The terminal window has lost focus. The user may be away.'
}

export function formatUserReturnedAfterAbsence(): string {
  return `The user has returned after being away. Subagents are actively running — consider surfacing a status update.`
}