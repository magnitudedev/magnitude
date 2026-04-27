/**
 * Canonical model identity — provider-independent.
 *
 * Every Model entry represents an open-source model with a known HuggingFace repo
 * and a fetchable chat template. Closed-source models do not have canonical entries.
 */

export type ModelId = string

export interface Model {
  readonly id: ModelId
  readonly name: string
  readonly family: string
  readonly hfRepo: string
  readonly template: string
}
