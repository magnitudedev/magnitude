import { Effect, Schema, Scope } from 'effect'
import type {
  ProviderId,
  ProviderModelId,
  SlotId,
} from '@magnitudedev/sdk'
import type { ModelRequestProgress, StreamStartFailure } from '@magnitudedev/ai'
import { ModelReleaseReasonSchema } from '@magnitudedev/protocol'

export class ModelRequestPreparationFailed extends Schema.TaggedError<ModelRequestPreparationFailed>()(
  'ModelRequestPreparationFailed',
  {
    code: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  },
) {}

export class ModelRequestPreparationCancelled extends Schema.TaggedError<ModelRequestPreparationCancelled>()(
  'ModelRequestPreparationCancelled',
  {
    reason: ModelReleaseReasonSchema,
  },
) {}

export type ModelRequestPreparationError =
  | ModelRequestPreparationFailed
  | ModelRequestPreparationCancelled

export type AgentModelStartFailure =
  | ModelRequestPreparationError
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
) => Effect.Effect<
  void,
  ModelRequestPreparationError,
  Scope.Scope
>
