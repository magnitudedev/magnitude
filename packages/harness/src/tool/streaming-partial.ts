/**
 * Streaming partial types and utilities for incremental tool input accumulation.
 */

/** A streaming leaf value — discriminated on finality. */
export type StreamingLeaf<T> =
  | { isFinal: true; value: T }
  | { isFinal: false; value: string }

/**
 * Transforms a tool's TInput into its streaming shape.
 * Fields arrive incrementally — scalars are StreamingLeaf, objects are partial, arrays accumulate.
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

/** All valid deep paths into T as tuple types. */
export type DeepPaths<T> = T extends object
  ? {
      [K in keyof T & string]: [K] | [K, ...DeepPaths<NonNullable<T[K]>>]
    }[keyof T & string]
  : never

/**
 * Apply a streaming text delta at the given path into a StreamingPartial.
 * Pure — navigates/creates nested structure, appends text at the leaf.
 */
export function applyFieldChunk<T>(
  partial: StreamingPartial<T>,
  path: readonly string[],
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

/**
 * Recursively unwrap StreamingLeaf wrappers from a streaming partial structure.
 * Returns plain values with all leaf wrappers removed.
 */
export function extractStreamingPartialValues(partial: unknown): unknown {
  if (partial === null || partial === undefined) return partial
  if (typeof partial !== "object") return partial

  if ("isFinal" in partial && "value" in partial) {
    return (partial as { readonly value: unknown }).value
  }

  if (Array.isArray(partial)) {
    return partial.map((item) => extractStreamingPartialValues(item))
  }

  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(partial)) {
    result[key] = extractStreamingPartialValues(value)
  }
  return result
}
