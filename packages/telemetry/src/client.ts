/**
 * PostHog Telemetry Client — singleton for anonymous usage tracking.
 *
 * All capture calls are wrapped in try/catch — telemetry should never
 * crash or interfere with the application.
 */

import { PostHog } from 'posthog-node'
import { logger } from '@magnitudedev/logger'
import { getOrCreateMachineId } from './machine-id'

const POSTHOG_PROJECT_TOKEN = 'phc_U46ctUD56sDoszSQRyj0OmjoPGOH5cJfFtkYzuOnmeu'
const POSTHOG_HOST = 'https://us.i.posthog.com'

let client: PostHog | null = null
let distinctId: string | null = null
let enabled = false

export interface TelemetryInitOptions {
  /** From config — default true. Set false to disable. */
  telemetryEnabled?: boolean
}

/**
 * Initialize the telemetry client.
 * Checks env var MAGNITUDE_TELEMETRY first, then options.telemetryEnabled.
 */
export function initTelemetry(options: TelemetryInitOptions = {}): void {
  // Env var takes precedence over config
  const envVal = process.env.MAGNITUDE_TELEMETRY
  if (envVal !== undefined) {
    const disabled = ['0', 'false', 'off', 'no'].includes(envVal.toLowerCase())
    if (disabled) {
      enabled = false
      logger.info('[Telemetry] Disabled via MAGNITUDE_TELEMETRY env var')
      return
    }
    // Env var is set to an enabling value — skip config check, proceed to init
  } else if (options.telemetryEnabled === false) {
    // Only check config if env var is NOT set
    enabled = false
    logger.info('[Telemetry] Disabled via config')
    return
  }

  try {
    distinctId = getOrCreateMachineId()
    client = new PostHog(POSTHOG_PROJECT_TOKEN, {
      host: POSTHOG_HOST,
      flushAt: 20,
      flushInterval: 10000,
      requestTimeout: 10000,
      disableGeoip: true,
    })
    enabled = true
    logger.info('[Telemetry] Initialized')
  } catch (e) {
    logger.warn({ error: e }, '[Telemetry] Failed to initialize, telemetry disabled')
    enabled = false
    client = null
  }
}

/**
 * Capture a telemetry event. No-op if telemetry is disabled.
 */
export function capture(event: string, properties: Record<string, unknown> = {}): void {
  if (!enabled || !client || !distinctId) return
  try {
    client.capture({
      distinctId,
      event,
      properties,
    })
  } catch {
    // Silently fail — telemetry should never crash the app
  }
}

/**
 * Check if telemetry is currently enabled.
 */
export function isTelemetryEnabled(): boolean {
  return enabled && client !== null
}

/**
 * Flush and shutdown the PostHog client.
 * Should be called before process exit. Times out after 3 seconds.
 */
export async function shutdownTelemetry(): Promise<void> {
  if (!client) return
  try {
    await Promise.race([
      client.shutdown(),
      new Promise<void>((resolve) => setTimeout(resolve, 3000)),
    ])
  } catch {
    // Non-fatal — process is exiting anyway
  }
  client = null
}
