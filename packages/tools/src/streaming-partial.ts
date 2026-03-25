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
      : T[K] extends Record<string, any>
        ? StreamingPartial<T[K]>
        : StreamingLeaf<T[K]>
}
