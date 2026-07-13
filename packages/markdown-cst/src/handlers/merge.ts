/**
 * Type-safe handler merging utilities.
 *
 * Provides compile-time checking for:
 * 1. No duplicate handlers across files
 * 2. All token types are covered (exhaustiveness)
 */

import type { Handlers } from '../types'

// =============================================================================
// DUPLICATE DETECTION TYPES
// =============================================================================

/**
 * Get the overlapping keys between two types.
 */
type OverlappingKeys<A, B> = Extract<keyof A, keyof B>

/**
 * Assert no overlap between two types. Returns the intersection if valid,
 * or an error type if there are duplicates.
 */
type MergeTwo<A, B> = OverlappingKeys<A, B> extends never
  ? A & B
  : { _error: 'Duplicate handler keys'; _duplicates: OverlappingKeys<A, B> }

/**
 * Recursively merge a tuple of partial handler objects.
 * Fails at compile time if any two objects share keys.
 */
export type MergePartials<T extends object[]> = T extends []
  ? {}
  : T extends [infer Only extends object]
    ? Only
    : T extends [infer First extends object, infer Second extends object]
      ? MergeTwo<First, Second>
      : T extends [infer First extends object, infer Second extends object, ...infer Rest extends object[]]
        ? MergeTwo<First, Second> extends { _error: string }
          ? MergeTwo<First, Second>
          : MergePartials<[MergeTwo<First, Second>, ...Rest]>
        : never

/**
 * Check if merged type is valid (no errors and covers all handlers).
 */
export type CheckMerge<T> = T extends { _error: string }
  ? T  // Propagate error
  : Exclude<keyof Handlers, keyof T> extends never
    ? Handlers  // All keys covered
    : { _error: 'Missing handlers'; _missing: Exclude<keyof Handlers, keyof T> }

// =============================================================================
// RUNTIME MERGE
// =============================================================================

/**
 * Merge partial handlers into a complete Handlers object.
 * Type parameter ensures no duplicates and full coverage.
 */
export function mergeHandlers<T extends Partial<Handlers>[]>(
  ...partials: T
): CheckMerge<MergePartials<T>> {
  const result = {} as Record<string, unknown>
  for (const partial of partials) {
    Object.assign(result, partial)
  }
  return result as CheckMerge<MergePartials<T>>
}
