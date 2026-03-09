/**
 * User presence prompt formatting.
 *
 * Formats window focus state changes as system notifications for the orchestrator.
 */

export function formatUserPresence(focused: boolean): string {
  return focused
    ? '<user-presence>The terminal window has regained focus. The user is likely present.</user-presence>'
    : '<user-presence>The terminal window has lost focus. The user may be away.</user-presence>'
}

export function formatUserReturnedAfterAbsence(): string {
  return `<user-presence confirmed="true">The user has returned after being away. Subagents are actively running — consider surfacing a status update.</user-presence>`
}