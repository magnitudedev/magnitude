/**
 * Input Builder — utilities for consumers to apply streaming field chunks.
 *
 * applyFieldChunk: pure function that applies a (path, text) delta to a
 * StreamingPartial<TInput>. State models call this in their reduce() to build
 * up the streaming input partial from ToolInputFieldChunk events.
 *
 * The parser handles all JSON parsing and field coercion. The engine does not
 * need to build input — ToolInputReady already carries the assembled input.
 * This module provides the consumer-side utility for streaming partials.
 */

import type { DeepPaths, StreamingPartial } from './types'

/**
 * Apply a field chunk delta to a streaming partial.
 *
 * Given a path (e.g. ["config", "host"]) and a text delta ("localhost"),
 * sets the value at that path in the partial. For scalar fields, the path
 * has one element and the value is the accumulated text. For JSON fields,
 * the path reflects the current nesting depth in the JSON structure.
 *
 * This is NOT re-parsing — the parser already computed the path. The consumer
 * just applies the delta at the given location.
 */
export function applyFieldChunk<TInput>(
  partial: StreamingPartial<TInput>,
  path: DeepPaths<TInput>,
  text: string,
): StreamingPartial<TInput> {
  if (!Array.isArray(path) || path.length === 0) return partial

  const result = shallowClone(partial) as Record<string, unknown>

  if (path.length === 1) {
    const key = path[0] as string
    const existing = result[key]
    // For leaf nodes, accumulate text
    result[key] = typeof existing === 'string' ? existing + text : text
    return result as StreamingPartial<TInput>
  }

  // Recurse into nested path
  const [head, ...tail] = path as [string, ...string[]]
  const nested = (result[head] ?? {}) as Record<string, unknown>
  result[head] = applyFieldChunk(
    nested as StreamingPartial<unknown>,
    tail as unknown as DeepPaths<unknown>,
    text,
  )
  return result as StreamingPartial<TInput>
}

function shallowClone<T>(obj: T): T {
  if (obj === null || obj === undefined) return {} as T
  if (typeof obj !== 'object') return obj
  return { ...obj }
}
