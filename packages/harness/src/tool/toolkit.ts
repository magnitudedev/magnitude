import type { Effect } from "effect"
import type { HarnessTool, HarnessToolConcrete } from "./tool"
import type { StateModel } from "./state-model"

// --- Helpers ---

/** Type-safe Object.keys — standard TS pattern, safe because T extends Record<string, ...> */
function typedKeys<T extends Record<string, unknown>>(obj: T): (keyof T & string)[] {
  return Object.keys(obj) as (keyof T & string)[]
}

// --- ToolkitEntry ---

export interface ToolkitEntry<
  TTool = HarnessTool,
  TStateModel = StateModel | undefined,
> {
  readonly tool: TTool
  readonly state?: TStateModel
}

// --- Toolkit ---

export interface Toolkit<T extends Record<string, ToolkitEntry> = Record<string, ToolkitEntry>> {
  readonly entries: T
  readonly keys: T extends infer U extends Record<string, ToolkitEntry> ? readonly (keyof U & string)[] : never
  pick<K extends string[]>(...keys: K): Toolkit<Pick<T, K[number] & keyof T>>
  omit<K extends string[]>(...keys: K): Toolkit<Omit<T, K[number] & keyof T>>
}

// --- Type helpers ---

export type ToolkitKeys<T extends Toolkit> = keyof T["entries"] & string

export type ToolkitTool<T extends Toolkit, K extends ToolkitKeys<T>> =
  T["entries"][K]["tool"]

export type ToolkitState<T extends Toolkit, K extends ToolkitKeys<T>> =
  T["entries"][K] extends { state: infer S } ? S : undefined

/** Extract the combined Effect R requirements from a tool's execute and stream onInput signatures. */
export type ToolRequirements<T> =
  T extends HarnessToolConcrete<any, any, any, any, infer RExecute, infer RStream, any>
    ? RExecute | RStream
    : T extends { readonly execute: (...args: infer _Args) => Effect.Effect<unknown, unknown, infer R> }
      ? R
      : never

export type ToolkitRequirements<T extends Toolkit> = {
  [K in ToolkitKeys<T>]: ToolRequirements<T["entries"][K]["tool"]>
}[ToolkitKeys<T>]

// --- Implementation ---

class ToolkitImpl<T extends Record<string, ToolkitEntry>> implements Toolkit<T> {
  readonly entries: T
  readonly keys: T extends infer U extends Record<string, ToolkitEntry> ? readonly (keyof U & string)[] : never

  constructor(entries: T) {
    this.entries = Object.freeze({ ...entries }) as T
    this.keys = typedKeys(entries) as any
  }

  pick<K extends string[]>(...keys: K): Toolkit<Pick<T, K[number] & keyof T>> {
    const picked: Record<string, ToolkitEntry> = {}
    for (const key of keys) {
      if (!(key in this.entries)) {
        throw new Error(`Toolkit.pick: key "${key}" not found. Available: ${(this.keys as readonly string[]).join(", ")}`)
      }
      picked[key] = this.entries[key as keyof T] as ToolkitEntry
    }
    return new ToolkitImpl(picked) as any
  }

  omit<K extends string[]>(...keys: K): Toolkit<Omit<T, K[number] & keyof T>> {
    const omitSet = new Set<string>(keys)
    const remaining: Record<string, ToolkitEntry> = {}
    for (const key of (this.keys as readonly string[])) {
      if (!omitSet.has(key)) {
        remaining[key] = this.entries[key as keyof T] as ToolkitEntry
      }
    }
    return new ToolkitImpl(remaining) as any
  }
}

// --- defineToolkit ---

export function defineToolkit<const T extends Record<string, ToolkitEntry>>(entries: T): Toolkit<T> {
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
