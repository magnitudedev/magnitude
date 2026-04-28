/**
 * applyFieldEnd — immutably set a value at a nested path within a JS value.
 *
 * Path encoding matches codec field events:
 *   - object keys:  string segments like "config", "tls"
 *   - array indices: decimal-string segments like "0", "1"
 *
 * When path is empty, returns `value` directly (replaces root).
 * Creates intermediate objects/arrays as needed.
 */
export function applyFieldEnd(
  input: unknown,
  path: readonly string[],
  value: unknown,
): unknown {
  if (path.length === 0) return value

  const [head, ...tail] = path
  const isNumeric = /^\d+$/.test(head)

  if (isNumeric) {
    const arr: unknown[] = Array.isArray(input) ? [...(input as unknown[])] : []
    const idx = Number(head)
    arr[idx] = applyFieldEnd(arr[idx], tail, value)
    return arr
  } else {
    const obj: Record<string, unknown> =
      input !== null && typeof input === 'object' && !Array.isArray(input)
        ? { ...(input as Record<string, unknown>) }
        : {}
    obj[head] = applyFieldEnd(obj[head], tail, value)
    return obj
  }
}
