import type { Effect } from "effect"
import type { HarnessTool } from "./tool"
import type { StateModel } from "./state-model"

// --- Helpers ---

/** Type-safe Object.keys — standard TS pattern, safe because T extends Record<string, ...> */
function typedKeys<T extends Record<string, unknown>>(obj: T): (keyof T & string)[] {
  return Object.keys(obj) as (keyof T & string)[]
}

// --- ToolkitEntry ---

// Base entry shape used as constraint — accepts both erased and concrete tools.
interface ToolkitEntryBase {
  readonly tool: { readonly definition: { readonly name: string } }
  readonly state?: StateModel | undefined
}

export interface ToolkitEntry<
  TTool = HarnessTool,
  TStateModel extends StateModel | undefined = StateModel | undefined,
> {
  readonly tool: TTool
  readonly state?: TStateModel
}

// --- Toolkit ---

export interface Toolkit<T extends Record<string, ToolkitEntryBase> = Record<string, ToolkitEntryBase>> {
  readonly entries: T
  readonly keys: readonly (keyof T & string)[]
  pick<K extends (keyof T & string)[]>(...keys: K): Toolkit<Pick<T, K[number]>>
  omit<K extends (keyof T & string)[]>(...keys: K): Toolkit<Omit<T, K[number]>>
}

// --- Type helpers ---

export type ToolkitKeys<T extends Toolkit> = keyof T["entries"] & string

export type ToolkitTool<T extends Toolkit, K extends ToolkitKeys<T>> =
  T["entries"][K]["tool"]

export type ToolkitState<T extends Toolkit, K extends ToolkitKeys<T>> =
  T["entries"][K] extends { state: infer S } ? S : undefined

/** Extract the Effect R channel from a tool's execute signature. Uses (...args: infer _Args) to avoid contravariance issues with concrete parameter types. */
export type ToolRequirements<T> =
  T extends { readonly execute: (...args: infer _Args) => Effect.Effect<unknown, unknown, infer R> } ? R : never

export type ToolkitRequirements<T extends Toolkit> = {
  [K in ToolkitKeys<T>]: ToolRequirements<T["entries"][K]["tool"]>
}[ToolkitKeys<T>]

// --- Implementation ---

class ToolkitImpl<T extends Record<string, ToolkitEntryBase>> implements Toolkit<T> {
  readonly entries: T
  readonly keys: readonly (keyof T & string)[]

  constructor(entries: T) {
    this.entries = Object.freeze({ ...entries }) as T
    this.keys = typedKeys(entries)
  }

  pick<K extends (keyof T & string)[]>(...keys: K): Toolkit<Pick<T, K[number]>> {
    const picked: Record<string, ToolkitEntryBase> = {}
    for (const key of keys) {
      if (!(key in this.entries)) {
        throw new Error(`Toolkit.pick: key "${key}" not found. Available: ${this.keys.join(", ")}`)
      }
      picked[key] = this.entries[key]
    }
    // TS can't narrow a dynamically-built record to Pick.
    return new ToolkitImpl(picked as Pick<T, K[number]>)
  }

  omit<K extends (keyof T & string)[]>(...keys: K): Toolkit<Omit<T, K[number]>> {
    const omitSet = new Set<string>(keys)
    const remaining: Record<string, ToolkitEntryBase> = {}
    for (const key of this.keys) {
      if (!omitSet.has(key)) {
        remaining[key] = this.entries[key]
      }
    }
    // TS can't narrow a dynamically-built record to Omit.
    return new ToolkitImpl(remaining as Omit<T, K[number]>)
  }
}

// --- defineToolkit ---

// Constraint uses a structural minimum to accept both erased and concrete tools.
// The `const T` inference captures the full concrete types.
export function defineToolkit<const T extends Record<string, { readonly tool: { readonly definition: { readonly name: string } }; readonly state?: StateModel }>>(entries: T): Toolkit<T> {
  return new ToolkitImpl(entries)
}

// --- mergeToolkits ---

type DisjointCheck<A, B> = Extract<keyof A, keyof B> extends never ? unknown : never

export function mergeToolkits<
  A extends Record<string, ToolkitEntry>,
  B extends Record<string, ToolkitEntry>,
>(
  a: Toolkit<A>,
  b: Toolkit<B>,
  ..._check: [DisjointCheck<A, B>] extends [never] ? ["Error: toolkits have overlapping keys"] : []
): Toolkit<A & B> {
  // Runtime collision check
  for (const key of b.keys) {
    if (key in a.entries) {
      throw new Error(`mergeToolkits: duplicate key "${key}" found in both toolkits`)
    }
  }
  // Disjoint keys verified at compile time (DisjointCheck) and runtime (collision check).
  return new ToolkitImpl({ ...a.entries, ...b.entries } as A & B)
}
