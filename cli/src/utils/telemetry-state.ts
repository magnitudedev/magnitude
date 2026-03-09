/**
 * Shared module-level state for telemetry.
 * Used by app.tsx to set the session tracker and by graceful-shutdown.ts to read it.
 * Exists as a separate module to avoid circular imports.
 */

import type { SessionTracker } from '@magnitudedev/telemetry'

let sessionTracker: SessionTracker | null = null

export function setSessionTracker(tracker: SessionTracker): void {
  sessionTracker = tracker
}

export function getSessionTracker(): SessionTracker | null {
  return sessionTracker
}
