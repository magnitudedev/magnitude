/**
 * ModelDriver — the interaction method for communicating with a provider's API.
 * Magnitude routes all providers through the BAML driver.
 */
export type ModelDriverId = 'baml'

export interface ModelDriver {
  readonly id: ModelDriverId
  readonly name: string
  readonly supportsStreaming: boolean
}

/** Known driver definitions */
export const DRIVERS: Record<ModelDriverId, ModelDriver> = {
  baml: { id: 'baml', name: 'BAML', supportsStreaming: true },
}