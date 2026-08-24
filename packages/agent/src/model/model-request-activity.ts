import type { IcnModelPreparation } from '@magnitudedev/sdk'
import { Ambient } from '@magnitudedev/event-core'

interface ModelRequestTurn {
  readonly turnId: string
  readonly chainId: string
  readonly forkId: string | null
}

export interface ModelRequestActivityObservation {
  readonly turn: ModelRequestTurn
  readonly activity: ModelRequestActivity
}

export type ModelRequestActivity =
  | { readonly _tag: 'Starting'; readonly requestId: string | null }
  | { readonly _tag: 'Preparing'; readonly preparation: IcnModelPreparation; readonly requestId: string | null }
  | { readonly _tag: 'Streaming'; readonly requestId: string | null }
  | { readonly _tag: 'Ended'; readonly requestId: string | null }

export const ModelRequestActivityAmbient =
  Ambient.define<ModelRequestActivityObservation | null>({
    name: 'ModelRequestActivity',
    initial: null,
  })
