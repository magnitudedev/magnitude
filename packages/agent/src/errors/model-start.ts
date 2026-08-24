import type { ModelAttemptFailureSnapshot, UpstreamRetryability as UpstreamRetryabilityType } from '@magnitudedev/ai'
import type { AttemptCommitPolicy, TurnOutcome } from '../events'
import type { AgentModelStartFailure } from '../model/model-request-preparation'
import type { ErrorPresentation } from './present'
import {
  finalizeModelAttemptFailure,
  formatModelAttemptFailure,
  modelAttemptRetryability,
  type ModelAttemptFinalizerDecision,
} from './model-attempt'

export interface AgentModelStartFinalizerDecision {
  readonly outcome: TurnOutcome
  readonly retry: ModelAttemptFinalizerDecision['retry']
  readonly commitPolicy: AttemptCommitPolicy
  readonly presentation: ErrorPresentation
  readonly snapshot: ModelAttemptFailureSnapshot
}

export function finalizeAgentModelStartFailure(input: {
  readonly failure: AgentModelStartFailure
  readonly retryCount: number
  readonly maxRetries: number
}): AgentModelStartFinalizerDecision {
  return finalizeModelAttemptFailure(input)
}

export function agentModelStartRetryability(
  failure: AgentModelStartFailure,
): UpstreamRetryabilityType {
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
  return formatModelAttemptFailure(failure)
}
