/**
 * Helper for defining partial handlers with full type safety.
 *
 * Usage:
 *   const blockEnter = definePartialHandlers({
 *     paragraph: (ctx, token) => { ... },
 *     atxHeading: (ctx, token) => { ... },
 *   })
 *
 * The handler functions get full type inference for ctx and token.
 */

import type { Handlers } from '../types'

/**
 * Define a subset of handlers with full type inference.
 * The returned object is typed as Pick<Handlers, K> where K is the keys you provide.
 */
export function definePartialHandlers<K extends keyof Handlers>(
  handlers: { [P in K]: Handlers[P] }
): { [P in K]: Handlers[P] } {
  return handlers
}
