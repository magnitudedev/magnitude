/**
 * ForkContext - Service providing current forkId during execution
 *
 * The agent layer is responsible for providing this context.
 * Tools and workers can access it to know which fork they're running in.
 */

import { Context } from 'effect'

/**
 * ForkContext service interface
 */
export interface ForkContextService {
  /** Current forkId - null means root agent */
  readonly forkId: string | null
}

/**
 * ForkContext tag for dependency injection
 */
export const ForkContext = Context.GenericTag<ForkContextService>('ForkContext')
