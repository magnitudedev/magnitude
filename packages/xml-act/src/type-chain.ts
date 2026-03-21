import type { ChildAcc } from '@magnitudedev/tools'

/**
 * Extract attribute names from a readonly array of { attr: string } objects.
 * Requires `const` type parameter on the mapping to preserve literal types.
 */
export type AttrNames<T> = T extends readonly { attr: infer A extends string }[]
  ? A
  : never

/**
 * Extract tag names from a readonly array of { tag: string } objects.
 */
export type ChildTagNames<T> = T extends readonly { tag: infer U extends string }[]
  ? U
  : never

/**
 * Extract tag names from children array (which may use `tag` or fall back to `field`).
 */
export type ChildrenTagNames<T> = T extends readonly (infer Item)[]
  ? Item extends { tag: infer U extends string }
    ? U
    : Item extends { field: infer F extends string }
      ? F
      : never
  : never

/**
 * Derive the streaming shape from an XML mapping config.
 * This maps the XML namespace (attrs, tags) to a StreamingInput shape.
 */
export type DeriveStreamingShape<TMapping> = {
  fields: TMapping extends { input: { attributes: infer A } }
    ? { [K in AttrNames<A>]?: string }
    : {}
  body: TMapping extends { input: { body: string } } ? string : ''
  children: TMapping extends { input: infer I }
    ? (I extends { childTags: infer CT }
        ? { [K in ChildTagNames<CT>]?: ChildAcc[] }
        : {}) &
      (I extends { children: infer CH }
        ? { [K in ChildrenTagNames<CH>]?: ChildAcc[] }
        : {}) &
      (I extends { childRecord: { tag: infer CR extends string } }
        ? { [K in CR]?: ChildAcc[] }
        : {})
    : {}
}

export type DeriveFields<TMapping> = DeriveStreamingShape<TMapping>['fields']
export type DeriveChildren<TMapping> = DeriveStreamingShape<TMapping>['children']
