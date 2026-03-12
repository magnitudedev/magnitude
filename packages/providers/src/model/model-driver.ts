/**
 * ModelDriver — the interaction method for communicating with a provider's API.
 * Examples: BAML (standard), OpenAI Responses API, direct HTTP.
 * Same provider can support multiple drivers.
 */
export type ModelDriverId = 'baml' | 'openai-responses'

export interface ModelDriver {
  readonly id: ModelDriverId
  readonly name: string
  readonly supportsStreaming: boolean
}

/** Known driver definitions */
export const DRIVERS: Record<ModelDriverId, ModelDriver> = {
  baml: { id: 'baml', name: 'BAML', supportsStreaming: true },
  'openai-responses': { id: 'openai-responses', name: 'OpenAI Responses API', supportsStreaming: true },
}