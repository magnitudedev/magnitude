/**
 * Error presentation.
 *
 * Single source of truth for what every TurnOutcome looks like to the user
 * (and what it looks like fed back to the model). All copy lives here.
 */

import type {
  TurnOutcome,
  ProviderNotReadyDetail,
  ConnectionFailureDetail,
  SafetyStopReason,
} from '../events'
import type { ModelAttemptFailureSnapshot } from '@magnitudedev/ai'

export type ErrorSurface = 'inline' | 'toast' | 'silent'
export type ErrorSeverity = 'error' | 'warning' | 'info'

/** Action IDs the CLI knows how to dispatch when the user invokes an action CTA. */
export type ActionId = 'open-settings' | 'open-usage'

export type ErrorCta =
  | { readonly kind: 'url'; readonly label: string; readonly url: string }
  | { readonly kind: 'action'; readonly label: string; readonly actionId: ActionId; readonly chord: string }

export interface ErrorPresentation {
  /** Where this should appear, if anywhere. */
  readonly surface: ErrorSurface
  readonly severity: ErrorSeverity
  /** User-facing copy. Empty when surface is 'silent'. */
  readonly message: string
  /** Optional CTA shown beneath the message. */
  readonly cta?: ErrorCta
  /** Text to feed back to the model as observation. Omit for none. */
  readonly llmFeedback?: string
  /** Whether the underlying condition is auto-retried by the agent loop. */
  readonly retryable: boolean
}

const CLOUD_USAGE_CTA: ErrorCta = { kind: 'url', label: 'View cloud usage', url: 'https://app.magnitude.dev/billing' }
const MAGNITUDE_PRO_CTA: ErrorCta = { kind: 'url', label: 'Start Magnitude Pro', url: 'https://app.magnitude.dev/billing' }
const UPDATE_MAGNITUDE_CTA: ErrorCta = { kind: 'url', label: 'Update Magnitude', url: 'https://docs.magnitude.dev/get-started' }
const OPEN_SETTINGS_CTA: ErrorCta = { kind: 'action', label: 'Open settings', actionId: 'open-settings', chord: 'ctrl+s' }

const SILENT: ErrorPresentation = {
  surface: 'silent',
  severity: 'info',
  message: '',
  retryable: false,
}

function presentProviderNotReady(detail: ProviderNotReadyDetail): ErrorPresentation {
  switch (detail._tag) {
    case 'AuthFailed':
      return {
        surface: 'inline',
        severity: 'error',
        message: 'Authentication failed. API key may be invalid or revoked.',
        cta: OPEN_SETTINGS_CTA,
        llmFeedback: 'Authentication failed. API key may be invalid or revoked.',
        retryable: false,
      }
    case 'OutOfSync':
      return {
        surface: 'inline',
        severity: 'error',
        message: 'Magnitude is out of sync with the server. Try updating to the latest version.',
        cta: UPDATE_MAGNITUDE_CTA,
        llmFeedback: 'Out-of-sync error from server. The CLI may need to be updated.',
        retryable: false,
      }
    case 'SubscriptionRequired':
      return {
        surface: 'inline',
        severity: 'error',
        message: detail.message,
        cta: MAGNITUDE_PRO_CTA,
        llmFeedback: detail.message,
        retryable: false,
      }
    case 'UsageLimitExceeded': {
      return {
        surface: 'inline',
        severity: 'error',
        message: detail.message,
        cta: CLOUD_USAGE_CTA,
        llmFeedback: detail.message,
        retryable: false,
      }
    }
  }
}

function presentConnectionFailure(detail: ConnectionFailureDetail): ErrorPresentation {
  // Connection failures are auto-retried by the loop. We render a dim status
  // indicator inside the think block (driven separately in display.ts) and
  // feed a short note back to the model. We do NOT render an inline error
  // for each retry — that's noise.
  const llmFeedback = `Model attempt failed transiently; retrying.\n${detail.failure.message}`
  return {
    surface: 'silent',
    severity: 'warning',
    message: '',
    llmFeedback,
    retryable: true,
  }
}

function presentSafetyStop(reason: SafetyStopReason): ErrorPresentation {
  let userMessage: string
  let feedback: string
  switch (reason._tag) {
    case 'IdenticalResponseCircuitBreaker':
      userMessage = `Stopped after ${reason.threshold} identical responses`
      feedback = `Safety stop: repeated identical responses reached threshold ${reason.threshold}.`
      break
    case 'Other':
      userMessage = reason.message
      feedback = `Safety stop: ${reason.message}`
      break
  }
  return {
    surface: 'inline',
    severity: 'error',
    message: userMessage,
    llmFeedback: feedback,
    retryable: false,
  }
}

function presentOverthinking(limit: number): ErrorPresentation {
  const message = `Your thinking exceeded the ${limit} character limit. Remember that thinking in isolation has no value. Keep thinking concise and keep yourself grounded with tools and workers. Do not repeat any thinking you already conducted.`
  return {
    surface: 'inline',
    severity: 'warning',
    message,
    llmFeedback: message,
    retryable: true,
  }
}

function presentStreamFailed(failure: ModelAttemptFailureSnapshot): ErrorPresentation {
  const message = failure.tag === 'StreamProviderCorrectnessViolation'
    ? 'Provider stream contract violation'
    : 'Model response stream failed'

  return {
    surface: 'inline',
    severity: 'error',
    message,
    llmFeedback: failure.message,
    retryable: false,
  }
}

/**
 * Map a TurnOutcome to its presentation.
 *
 * Pure function. Every variant has a single defined behavior here — change
 * copy or routing in one place and every surface picks it up.
 */
export function present(outcome: TurnOutcome): ErrorPresentation {
  switch (outcome._tag) {
    case 'Completed':
      return SILENT
    case 'Cancelled':
      return { ...SILENT, llmFeedback: 'interrupted' }
    case 'ContextWindowExceeded':
      return {
        surface: 'silent',
        severity: 'warning',
        message: '',
        llmFeedback: 'Context window exceeded; waiting for compaction or context reduction.',
        retryable: false,
      }
    case 'OutputTruncated':
      return {
        surface: 'inline',
        severity: 'error',
        message: 'Response exceeded output limit',
        llmFeedback: 'Output was truncated. Respond in smaller, more bounded steps.',
        retryable: false,
      }
    case 'SafetyStop':
      return presentSafetyStop(outcome.reason)
    case 'ProviderNotReady':
      return presentProviderNotReady(outcome.detail)
    case 'ConnectionFailure':
      return presentConnectionFailure(outcome.detail)
    case 'StreamFailed':
      return presentStreamFailed(outcome.failure)
    case 'UnexpectedError':
      return {
        surface: 'inline',
        severity: 'error',
        message: outcome.detail.message,
        llmFeedback: outcome.detail.message,
        retryable: false,
      }
    case 'Overthinking':
      return presentOverthinking(outcome.limit)
    case 'ToolInputValidationFailure':
      return SILENT
    case 'ToolExecutionError':
      return SILENT
    case 'GateRejected':
      return SILENT
  }
}
