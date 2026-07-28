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

export interface ModelRequestPreparationCancellationSnapshot {
  readonly tag: 'ModelRequestPreparationCancelled'
  readonly reason: Extract<
    AgentModelStartFailure,
    { readonly _tag: 'ModelRequestPreparationCancelled' }
  >['reason']
}

export interface AgentModelStartFinalizerDecision {
  readonly outcome: TurnOutcome
  readonly retry: ModelAttemptFinalizerDecision['retry']
  readonly commitPolicy: AttemptCommitPolicy
  readonly presentation: ErrorPresentation
  readonly snapshot:
    | ModelAttemptFailureSnapshot
    | ModelRequestPreparationFailureSnapshot
    | ModelRequestPreparationCancellationSnapshot
}

const isModelRequestPreparationFailure = (
  failure: AgentModelStartFailure,
): failure is ModelRequestPreparationFailed =>
  failure._tag === 'ModelRequestPreparationFailed'

const isModelRequestPreparationCancellation = (
  failure: AgentModelStartFailure,
): failure is Extract<
  AgentModelStartFailure,
  { readonly _tag: 'ModelRequestPreparationCancelled' }
> => failure._tag === 'ModelRequestPreparationCancelled'

export function finalizeAgentModelStartFailure(input: {
  readonly failure: AgentModelStartFailure
  readonly retryCount: number
  readonly maxRetries: number
}): AgentModelStartFinalizerDecision {
  const failure = input.failure
  if (failure._tag === 'ModelRequestPreparationCancelled') {
    const outcome: TurnOutcome = {
      _tag: 'Cancelled',
      reason: { _tag: 'ModelStopped', reason: failure.reason },
      requestId: null,
    }
    return {
      outcome,
      retry: { _tag: 'none' },
      commitPolicy: { _tag: 'commitErrorOnly' },
      presentation: present(outcome),
      snapshot: {
        tag: 'ModelRequestPreparationCancelled',
        reason: failure.reason,
      },
    }
  }
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
  if (isModelRequestPreparationFailure(failure)
    || isModelRequestPreparationCancellation(failure)) {
    return UpstreamRetryability.UpstreamNotRetryable({ reason: 'model_unavailable' })
  }
  return modelAttemptRetryability(failure)
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
  if (isModelRequestPreparationFailure(failure)) return failure.message
  if (isModelRequestPreparationCancellation(failure)) return 'Model loading was cancelled'
  return formatModelAttemptFailure(failure)
}
