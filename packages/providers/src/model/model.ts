import type { ModelId } from './canonical-model'

export interface ModelCosts {
  readonly inputPerM: number
  readonly outputPerM: number
  readonly cacheReadPerM: number | null
  readonly cacheWritePerM: number | null
}

export interface ProviderModel {
  readonly id: string
  readonly providerId: string
  readonly providerName: string
  readonly modelId: ModelId | null
  readonly name: string
  readonly contextWindow: number
  readonly maxContextTokens: number | null
  readonly maxOutputTokens: number | null
  readonly supportsToolCalls: boolean
  readonly supportsReasoning: boolean
  readonly supportsVision: boolean
  readonly supportsGrammar?: boolean
  /** Which inference paradigm this model entry uses. Defaults to 'xml-act'. */
  readonly paradigm?: 'xml-act' | 'native' | 'completions'
  readonly costs: ModelCosts | null
  readonly releaseDate?: string
  readonly discovery?: {
    primarySource: 'static' | 'models.dev' | 'openrouter-api'
    fetchedAt?: string
  }
}