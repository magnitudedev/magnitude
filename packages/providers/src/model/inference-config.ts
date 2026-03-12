/**
 * InferenceConfig — parameters that control how inference is performed.
 * Has defaults per model/slot, overridable per-request.
 */
export interface InferenceConfig {
  readonly temperature?: number
  readonly topP?: number
  readonly maxTokens?: number
  readonly stopSequences?: readonly string[]
}