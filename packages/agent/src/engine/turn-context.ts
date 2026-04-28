/**
 * TurnContextService — provides current turn metadata to tools during execution.
 *
 * Cortex provides this layer around each turn so tools can access turnId.
 */

import { Context } from 'effect'

export interface TurnContextService {
  readonly turnId: string
  readonly forkId: string | null
}

export const TurnContextTag = Context.GenericTag<TurnContextService>('TurnContext')
