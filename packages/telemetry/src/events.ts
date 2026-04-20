/**
 * Telemetry Event Helpers — typed capture functions for each telemetry event.
 *
 * Each function maps to a PostHog event with defined properties.
 * All properties are anonymous aggregate data — no personal content.
 */

import { capture } from './client'
import type { ModelUsage } from './session-tracker'

// ========== Session Lifecycle ==========

export function trackSessionStart(props: {
  platform: string
  shell: string
  isResume: boolean
}): void {
  capture('session_start', props)
}

export function trackSessionEnd(props: {
  durationSeconds: number
  totalTurns: number
  totalUserMessages: number
  totalInputTokens: number
  totalOutputTokens: number
  modelsUsed: ModelUsage[]
  compactionCount: number
}): void {
  capture('session_end', props)
}

// ========== Provider ==========

export function trackProviderConnected(props: {
  providerId: string
  authType: string
}): void {
  capture('provider_connected', props)
}
