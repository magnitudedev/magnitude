import { Data } from 'effect'

export interface ModelCosts {
  readonly inputPerM: number
  readonly outputPerM: number
  readonly cacheReadPerM: number | null
  readonly cacheWritePerM: number | null
}

export class Model extends Data.Class<{
  readonly id: string
  readonly providerId: string
  readonly name: string
  readonly contextWindow: number
  readonly maxOutputTokens: number | null
  readonly costs: ModelCosts | null
}> {}