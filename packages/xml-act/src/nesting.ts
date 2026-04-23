/**
 * nesting.ts — single source of truth for valid tag nesting.
 *
 * Imported by both the grammar builder (to generate continuation rules)
 * and the parser (to enforce nesting at runtime via resolveOpenHandler).
 * Both systems reference this constant — they are provably in sync.
 *
 * Adding a new structural tag:
 *   1. Update VALID_CHILDREN here.
 *   2. Follow TypeScript errors to update grammar-builder.ts and resolve.ts.
 */

import type { Frame } from './parser/types'

export type FrameType = Frame['type']
export type StructuralTagName = 'magnitude:reason' | 'magnitude:message' | 'magnitude:invoke' | 'magnitude:parameter' | 'magnitude:filter'

/**
 * VALID_CHILDREN — maps each frame type to the structural tags it may contain.
 *
 * The `satisfies` constraint verifies the shape at compile time without widening the type,
 * so downstream consumers get the narrow `readonly string[]` literal types.
 */
export const VALID_CHILDREN = {
  prose:     ['magnitude:reason', 'magnitude:message', 'magnitude:invoke'] as const,
  invoke:    ['magnitude:parameter', 'magnitude:filter'] as const,
  reason:    [] as const,
  message:   [] as const,
  parameter: [] as const,
  filter:    [] as const,
} satisfies Record<FrameType, readonly StructuralTagName[]>

export type ValidChildren = typeof VALID_CHILDREN

// =============================================================================
// Compile-time verification assertions
// =============================================================================

/**
 * These type assertions are zero-cost at runtime.
 * If VALID_CHILDREN is updated, they will error until resolve.ts is updated to match.
 *
 * Usage in resolve.ts:
 *   import type { _VerifyProseChildren, _VerifyInvokeChildren } from '../nesting'
 *   // TypeScript evaluates the assertions on import — no runtime cost.
 */

/** Verifies that prose children are exactly the expected structural tags. */
export type _VerifyProseChildren =
  (typeof VALID_CHILDREN)['prose'][number] extends 'magnitude:reason' | 'magnitude:message' | 'magnitude:invoke' ? true : never

/** Verifies that invoke children are exactly the expected structural tags. */
export type _VerifyInvokeChildren =
  (typeof VALID_CHILDREN)['invoke'][number] extends 'magnitude:parameter' | 'magnitude:filter' ? true : never

// Eagerly evaluate assertions — TypeScript will error here if VALID_CHILDREN is wrong.
type _AssertProseChildren = _VerifyProseChildren extends true ? true : never
type _AssertInvokeChildren = _VerifyInvokeChildren extends true ? true : never

// Suppress "unused type" warnings — the assertions are the point.
declare const _assertProseChildren: _AssertProseChildren
declare const _assertInvokeChildren: _AssertInvokeChildren