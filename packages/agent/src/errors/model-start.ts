import {
  UpstreamRetryability,
  type ModelAttemptFailureSnapshot,
  type UpstreamRetryability as UpstreamRetryabilityType,
} from '@magnitudedev/ai'
import type { AttemptCommitPolicy, TurnOutcome } from '../events'
import type {
  AgentModelStartFailure,
  ModelRequestPreparationFailed,
} from '../model/model-request-preparation'
import { present, type ErrorPresentation } from './present'
import {
  finalizeModelAttemptFailure,
  formatModelAttemptFailure,
  modelAttemptRetryability,
  type ModelAttemptFinalizerDecision,
} from './model-attempt'

export interface ModelRequestPreparationFailureSnapshot {
  readonly tag: 'ModelRequestPreparationFailed'
  readonly code: string
  readonly message: string
  readonly retryable: boolean
}

export interface AgentModelStartFinalizerDecision {
  readonly outcome: TurnOutcome
  readonly retry: ModelAttemptFinalizerDecision['retry']
  readonly commitPolicy: AttemptCommitPolicy
  readonly presentation: ErrorPresentation
  readonly snapshot: ModelAttemptFailureSnapshot | ModelRequestPreparationFailureSnapshot
}

const isModelRequestPreparationFailure = (
  failure: AgentModelStartFailure,
): failure is ModelRequestPreparationFailed =>
  failure._tag === 'ModelRequestPreparationFailed'

export function finalizeAgentModelStartFailure(input: {
  readonly failure: AgentModelStartFailure
  readonly retryCount: number
  readonly maxRetries: number
}): AgentModelStartFinalizerDecision {
  const failure = input.failure
  if (failure._tag !== 'ModelRequestPreparationFailed') {
    return finalizeModelAttemptFailure({
      failure,
      retryCount: input.retryCount,
      maxRetries: input.maxRetries,
    })
  }

  const snapshot: ModelRequestPreparationFailureSnapshot = {
    tag: 'ModelRequestPreparationFailed',
    code: failure.code,
    message: failure.message,
    retryable: failure.retryable,
  }
  const outcome: TurnOutcome = {
    _tag: 'ModelNotReady',
    failure: {
      code: failure.code,
      message: failure.message,
      retryable: failure.retryable,
    },
    requestId: null,
  }
  return {
    outcome,
    retry: { _tag: 'none' },
    commitPolicy: { _tag: 'commitErrorOnly' },
    presentation: present(outcome),
    snapshot,
  }
}

export function agentModelStartRetryability(
  failure: AgentModelStartFailure,
): UpstreamRetryabilityType {
  return isModelRequestPreparationFailure(failure)
    ? UpstreamRetryability.UpstreamNotRetryable({ reason: 'model_unavailable' })
    : modelAttemptRetryability(failure)
}

export function presentAgentModelStartFailure(
  failure: AgentModelStartFailure,
): ErrorPresentation {
  return finalizeAgentModelStartFailure({
    failure,
    retryCount: 0,
    maxRetries: 0,
  }).presentation
}

export function formatAgentModelStartFailure(
  failure: AgentModelStartFailure,
): string {
  return isModelRequestPreparationFailure(failure)
    ? failure.message
    : formatModelAttemptFailure(failure)
}
