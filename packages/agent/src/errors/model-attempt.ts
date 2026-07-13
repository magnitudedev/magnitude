import { Option } from 'effect'
import {
  defaultRetryabilityForFailure,
  formatModelAttemptFailureMessage,
  snapshotModelAttemptFailure,
  type ModelAttemptFailure,
  type ModelAttemptFailureSnapshot,
  type RetryAfter as RetryAfterType,
  type StreamStartFailure,
  type UpstreamRetryability,
} from '@magnitudedev/ai'
import type { AttemptCommitPolicy, TurnOutcome } from '../events'
import { TERMINAL_RETRY_EXHAUSTED_MESSAGE } from '../util/retry-backoff'
import { present, type ErrorPresentation } from './present'

export type AgentModelAttemptFailure = ModelAttemptFailure
export type AgentStreamStartFailure = StreamStartFailure

export interface ModelAttemptFinalizerInput {
  readonly failure: AgentModelAttemptFailure
  readonly retryCount: number
  readonly maxRetries: number
}

export interface ModelAttemptFinalizerDecision {
  readonly outcome: TurnOutcome
  readonly retry: { readonly _tag: 'retry'; readonly notBeforeMs: number | null } | { readonly _tag: 'none' }
  readonly commitPolicy: AttemptCommitPolicy
  readonly presentation: ErrorPresentation
  readonly snapshot: ModelAttemptFailureSnapshot
}

export function finalizeModelAttemptFailure(input: ModelAttemptFinalizerInput): ModelAttemptFinalizerDecision {
  const snapshot = snapshotModelAttemptFailure(input.failure)
  const retryability = defaultRetryabilityForFailure(input.failure)
  const requestId = requestIdFromFailure(input.failure)
  const providerOutcome = providerSpecificOutcome(input.failure, requestId)

  let outcome: TurnOutcome
  let retry: ModelAttemptFinalizerDecision['retry'] = { _tag: 'none' }
  let commitPolicy: AttemptCommitPolicy = { _tag: 'commitErrorOnly' }

  if (providerOutcome !== null) {
    outcome = providerOutcome
  } else if (retryability._tag === 'UpstreamRetryable') {
    if (input.retryCount >= input.maxRetries) {
      outcome = {
        _tag: 'UnexpectedError',
        detail: {
          _tag: 'Unknown',
          message: `${TERMINAL_RETRY_EXHAUSTED_MESSAGE}\n${snapshot.message}`,
        },
        requestId,
      }
    } else {
      const retryAfterMs = retryAfterToMs(retryability.retryAfter)
      outcome = {
        _tag: 'ConnectionFailure',
        detail: { _tag: 'ModelAttemptFailure', failure: snapshot },
        requestId,
      }
      retry = { _tag: 'retry', notBeforeMs: retryAfterMs }
      commitPolicy = { _tag: 'discardPartialAssistant' }
    }
  } else {
    switch (input.failure._tag) {
      case 'StreamStartProviderCorrectnessViolation':
        outcome = {
          _tag: 'UnexpectedError',
          detail: { _tag: 'ProviderDefect', message: snapshot.message },
          requestId,
        }
        break
      case 'StreamStartClientCorrectnessViolation':
      case 'StreamClientCorrectnessViolation':
        outcome = {
          _tag: 'UnexpectedError',
          detail: { _tag: 'EngineDefect', message: snapshot.message },
          requestId,
        }
        break
      case 'StreamProviderCorrectnessViolation':
      case 'StreamProviderError':
      case 'StreamOperationalFailure':
        outcome = { _tag: 'StreamFailed', failure: snapshot, requestId }
        break
      case 'StreamStartProviderRejection':
      case 'StreamStartOperationalFailure':
        outcome = {
          _tag: 'UnexpectedError',
          detail: { _tag: 'Unknown', message: snapshot.message },
          requestId,
        }
        break
    }
  }

  return {
    outcome,
    retry,
    commitPolicy,
    presentation: present(outcome),
    snapshot,
  }
}

export function modelAttemptRetryability(failure: AgentModelAttemptFailure): UpstreamRetryability {
  return defaultRetryabilityForFailure(failure)
}

export function presentModelAttemptFailure(failure: AgentModelAttemptFailure): ErrorPresentation {
  return finalizeModelAttemptFailure({ failure, retryCount: 0, maxRetries: 0 }).presentation
}

export function formatModelAttemptFailure(failure: AgentModelAttemptFailure): string {
  return formatModelAttemptFailureMessage(failure)
}

function requestIdFromFailure(failure: AgentModelAttemptFailure): string | null {
  return 'response' in failure && failure.response !== null
    ? failure.response.requestId
    : null
}

function providerSpecificOutcome(failure: AgentModelAttemptFailure, requestId: string | null): TurnOutcome | null {
  if (failure._tag !== 'StreamStartProviderRejection') return null

  switch (failure.rejection._tag) {
    case 'ContextLimitExceeded':
      return { _tag: 'ContextWindowExceeded', requestId }
    case 'AuthRejected':
      return { _tag: 'ProviderNotReady', detail: { _tag: 'AuthFailed' }, requestId }
    case 'InsufficientCredits':
      return {
        _tag: 'ProviderNotReady',
        detail: {
          _tag: 'InsufficientCredits',
          message: failure.rejection.message,
          balanceCents: failure.rejection.balanceCents,
        },
        requestId,
      }
    case 'ModelUnavailable':
    case 'ModelCapabilityMissing':
    case 'ProviderCapabilityMissing':
      return { _tag: 'ProviderNotReady', detail: { _tag: 'OutOfSync' }, requestId }
    case 'RateLimited':
    case 'UpstreamFailure':
      return null
    case 'ProviderInvariantViolation':
      return {
        _tag: 'UnexpectedError',
        detail: { _tag: 'ProviderDefect', message: formatModelAttemptFailureMessage(failure) },
        requestId,
      }
    case 'InvalidRequest':
      return {
        _tag: 'UnexpectedError',
        detail: { _tag: 'EngineDefect', message: formatModelAttemptFailureMessage(failure) },
        requestId,
      }
    default:
      return null
  }
}

function retryAfterToMs(retryAfter: RetryAfterType): number | null {
  return retryAfter._tag === 'RetryAfterMs' ? retryAfter.ms : null
}
