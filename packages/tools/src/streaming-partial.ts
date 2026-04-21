/**
 * A streaming leaf value — discriminated union on finality.
 * 
 * When `isFinal: true`, value has the true schema type (coerced).
 * When `isFinal: false`, value is a raw partial string (not yet coerced).
 * 
 * Examples:
 * - A Schema.Number field mid-stream: `{ isFinal: false, value: "12" }`
 * - A Schema.Number field finalized: `{ isFinal: true, value: 123 }`
 * - A Schema.Literal('foo') mid-stream: `{ isFinal: false, value: "fo" }`
 * - A Schema.Literal('foo') finalized: `{ isFinal: true, value: "foo" }`
 */
export type StreamingLeaf<T> =
  | { isFinal: true; value: T }
  | { isFinal: false; value: string }

/**
 * Transforms a tool's TInput schema type into its streaming shape.
 * 
 * During XML streaming:
 * - Fields arrive incrementally (any field may be absent)
 * - Scalar leaves are `StreamingLeaf<T>` — discriminated on finality
 * - Nested objects are partially populated
 * - Array elements accumulate incrementally; each element may be partial
 */
export type StreamingPartial<T> = {
  [K in keyof T]?: T[K] extends ReadonlyArray<infer E>
    ? Array<StreamingPartial<E>>
    : T[K] extends Array<infer E>
      ? Array<StreamingPartial<E>>
      : T[K] extends Record<string, unknown>
        ? StreamingPartial<T[K]>
        : StreamingLeaf<T[K]>
}

/**
 * All valid deep paths into T as tuple types.
 * e.g. for { a: { b: string } }: ["a"] | ["a", "b"]
 */
export type DeepPaths<T> = T extends object
  ? {
      [K in keyof T & string]: [K] | [K, ...DeepPaths<NonNullable<T[K]>>]
    }[keyof T & string]
  : never

/**
 * Apply a streaming text delta at the given path into a StreamingPartial.
 * Pure function — navigates/creates nested structure, appends text at the leaf.
 *
 * The parser computes the path (including nested JSON paths); the consumer
 * calls this in reduce() to build up their own StreamingPartial<TInput>.
 */
export function applyFieldChunk<T>(
  partial: StreamingPartial<T>,
  path: string[],
  delta: string,
): StreamingPartial<T> {
  if (path.length === 0) return partial

  const result = { ...partial } as Record<string, unknown>
  let cursor = result
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i]
    cursor[key] = cursor[key] ? { ...(cursor[key] as object) } : {}
    cursor = cursor[key] as Record<string, unknown>
  }

  const leaf = path[path.length - 1]
  const existing = cursor[leaf] as { isFinal: boolean; value: string } | undefined
  if (existing && !existing.isFinal) {
    cursor[leaf] = { isFinal: false, value: existing.value + delta }
  } else {
    cursor[leaf] = { isFinal: false, value: delta }
  }

  return result as StreamingPartial<T>
}
