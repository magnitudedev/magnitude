/**
 * Last attached session id — read by the graceful-shutdown handler to print
 * the resume hint. A plain module variable because the shutdown path runs
 * outside the atom registry.
 */
let lastSessionId: string | null = null

export function setLastSessionId(sessionId: string | null): void {
  lastSessionId = sessionId
}

export function getLastSessionId(): string | null {
  return lastSessionId
}
