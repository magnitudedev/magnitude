// ---------------------------------------------------------------------------
// OptionDef — typed, composable option-to-wire mapping
// ---------------------------------------------------------------------------

/**
 * Erased form — used in acceptance positions like `Record<string, OptionDef>`.
 * `any` in map parameter is required: contravariant acceptance position (ts-generics Pattern 6).
 */
export interface OptionDefErased {
  readonly _tag: "OptionDef"
  readonly required: boolean
  readonly default?: unknown
  readonly map: (value: any) => Record<string, unknown>
}

/**
 * Concrete form — carries full type information for a single option.
 */
export interface OptionDefConcrete<TValue, TWireReq, TRequired extends boolean> {
  readonly _tag: "OptionDef"
  readonly required: TRequired
  readonly default?: TValue
  readonly map: (value: TValue) => Partial<TWireReq>
}

/**
 * Never-switched union: bare `OptionDef` resolves to the erased form,
 * while `OptionDef<V, W, R>` resolves to the concrete form.
 */
export type OptionDef<TValue = never, TWireReq = never, TRequired extends boolean = false> =
  [TValue] extends [never] ? OptionDefErased : OptionDefConcrete<TValue, TWireReq, TRequired>

// ---------------------------------------------------------------------------
// Type-level utilities
// ---------------------------------------------------------------------------

/** Extract the value type from a concrete OptionDef. */
export type ExtractValue<T> = T extends OptionDefConcrete<infer V, any, any> ? V : never

/** Extract the required flag from a concrete OptionDef. */
export type ExtractRequired<T> = T extends OptionDefConcrete<any, any, infer R> ? R : false

/** Keys in an option record whose OptionDefs are required. */
export type RequiredKeys<T extends Record<string, OptionDef>> = {
  [K in keyof T]: ExtractRequired<T[K]> extends true ? K : never
}[keyof T]

/** Keys in an option record whose OptionDefs are optional. */
export type OptionalKeys<T extends Record<string, OptionDef>> = {
  [K in keyof T]: ExtractRequired<T[K]> extends true ? never : K
}[keyof T]

/** Infer the call-site options type from a record of OptionDefs. */
export type InferCallOptions<T extends Record<string, OptionDef>> =
  & { readonly [K in RequiredKeys<T>]: ExtractValue<T[K]> }
  & { readonly [K in OptionalKeys<T>]?: ExtractValue<T[K]> }

// ---------------------------------------------------------------------------
// Option namespace — factory functions
// ---------------------------------------------------------------------------

export const Option = {
  /**
   * Define an optional OptionDef.
   */
  define: <TValue, TWireReq>(
    map: (value: TValue) => Partial<TWireReq>,
    defaultValue?: TValue,
  ): OptionDef<TValue, TWireReq, false> => ({
    _tag: "OptionDef" as const,
    required: false as const,
    default: defaultValue,
    map,
  }),

  /**
   * Define a required OptionDef.
   */
  required: <TValue, TWireReq>(
    map: (value: TValue) => Partial<TWireReq>
  ): OptionDef<TValue, TWireReq, true> => ({
    _tag: "OptionDef" as const,
    required: true as const,
    map,
  }),
} as const

// ---------------------------------------------------------------------------
// applyOptionDefs — internal utility for dynamic option mapping
// ---------------------------------------------------------------------------

/**
 * Apply a record of erased OptionDefs to an options object, producing wire fragments.
 *
 * The `as Record<string, unknown>` cast is a widening cast backed by the
 * `T extends object` constraint (ts-generics Principle 3: constraint + cast = safe).
 */
export function applyOptionDefs<T extends object>(
  defs: Record<string, OptionDefErased>,
  options: T,
): Record<string, unknown> {
  const opts = options as Record<string, unknown>
  const result: Record<string, unknown> = {}
  for (const [key, def] of Object.entries(defs)) {
    const val = opts[key] ?? def.default
    if (val !== undefined) {
      Object.assign(result, def.map(val))
    }
  }
  return result
}
