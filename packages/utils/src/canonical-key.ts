/**
 * Canonical structural keys.
 *
 * `Query`, `Mutation`, and `Subscription` identity requires keys with Effect
 * `Equal` semantics. `canonical` turns any JSON-like value (including values
 * with `toJSON`, such as `Option` and `Date`) into a deterministic string:
 * object keys are sorted, `undefined` members are omitted, arrays keep order.
 * Two inputs that encode to the same structure therefore share one cache entry.
 */

interface WithToJson {
  readonly toJSON?: () => unknown
}

const canonicalize = (value: unknown): string | undefined => {
  if (value === undefined || typeof value === "function" || typeof value === "symbol") return undefined
  if (value === null) return "null"
  if (typeof value === "bigint") return JSON.stringify(value.toString())
  if (typeof value !== "object") return JSON.stringify(value)
  const withToJson: WithToJson = value
  if (typeof withToJson.toJSON === "function") return canonicalize(withToJson.toJSON())
  if (Array.isArray(value)) {
    const items: ReadonlyArray<unknown> = value
    return `[${items.map((item) => canonicalize(item) ?? "null").join(",")}]`
  }
  const entries: ReadonlyArray<readonly [string, unknown]> = Object.entries(value)
  const members = entries
    .map(([key, member]) => [key, canonicalize(member)] as const)
    .filter((entry): entry is readonly [string, string] => entry[1] !== undefined)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, member]) => `${JSON.stringify(key)}:${member}`)
  return `{${members.join(",")}}`
}

/** Deterministic string key for a JSON-like value. */
export const canonical = (value: unknown): string => canonicalize(value) ?? "null"
