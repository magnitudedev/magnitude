import { Effect, Schema, Scope } from 'effect'
import type {
  ProviderId,
  ProviderModelId,
  SlotId,
} from '@magnitudedev/sdk'
import type { ModelRequestProgress, StreamStartFailure } from '@magnitudedev/ai'

export class ModelRequestPreparationFailed extends Schema.TaggedError<ModelRequestPreparationFailed>()(
  'ModelRequestPreparationFailed',
  {
    code: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  },
) {}

export type AgentModelStartFailure =
  | ModelRequestPreparationFailed
  | StreamStartFailure

export interface ModelRequestPreparationInput {
  readonly slotId: SlotId
  readonly providerId: ProviderId
  readonly providerModelId: ProviderModelId
  readonly reportProgress: (
    progress: ModelRequestProgress,
  ) => Effect.Effect<void>
}

export type PrepareModelRequest = (
  input: ModelRequestPreparationInput,
) => Effect.Effect<void, ModelRequestPreparationFailed, Scope.Scope>
