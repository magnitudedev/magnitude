import type { ToolDefinition } from './tool-definition'

/** Base constraint: every catalog entry must have a tool */
export type BaseCatalogEntry = { tool: ToolDefinition }

/**
 * Generic typed container for tool entries.
 * Entry shape is inferred via const generic — agent can add binding, state, etc.
 */
/** Stored catalog type — covariant in T for safe assignment to bare ToolCatalog */
export interface ToolCatalog<out T extends Record<string, BaseCatalogEntry> = Record<string, BaseCatalogEntry>> {
  readonly entries: T
  readonly keys: readonly string[]
}

/** Concrete catalog with pick() — returned by defineCatalog */
export interface PickableCatalog<T extends Record<string, BaseCatalogEntry>> extends ToolCatalog<T> {
  pick<K extends (keyof T & string)[]>(...keys: K): PickableCatalog<Pick<T, K[number]>>
}

/** Utility types */
export type CatalogKeys<C> = C extends ToolCatalog<infer T> ? keyof T & string : never
export type CatalogEntry<C, K extends string> = C extends ToolCatalog<infer T> ? K extends keyof T ? T[K] : never : never
export type CatalogTool<C, K extends string> = CatalogEntry<C, K> extends { tool: infer TTool } ? TTool : never

function pickEntries<
  T extends Record<string, BaseCatalogEntry>,
  K extends readonly (keyof T & string)[]
>(entries: T, keys: K): Pick<T, K[number]> {
  const picked = {} as Pick<T, K[number]>
  for (const k of keys) picked[k] = entries[k]
  return picked
}

export function defineCatalog<const T extends Record<string, BaseCatalogEntry>>(entries: T): PickableCatalog<T> {
  const keyList = Object.keys(entries) as (keyof T & string)[]
  return {
    entries,
    keys: keyList,
    pick(...keys) {
      return defineCatalog(pickEntries(entries, keys))
    },
  }
}
