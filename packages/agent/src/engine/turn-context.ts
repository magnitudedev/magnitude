/**
 * TurnContextService — provides current turn metadata to tools during execution.
 *
 * Cortex provides this layer around each turn so tools can access turn identity.
 */

import { Context } from 'effect'

export interface TurnContextService {
  readonly turnId: string
  readonly chainId: string
  readonly forkId: string | null
}

export const TurnContextTag = Context.GenericTag<TurnContextService>('TurnContext')
