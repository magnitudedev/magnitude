import { Schema, Option } from 'effect'

// ---------------------------------------------------------------------------
// Field-level safety check — operates on the decoded struct Type
// ---------------------------------------------------------------------------

/**
 * A field is unsafe if its decoded type (as accessed from the struct's
 * `Schema.Schema.Type<S>`) includes `undefined` AND that type is NOT an
 * `Option`.
 *
 * - Required field (`T = X`): no `undefined` → safe.
 * - `optionalWith(X, { as: "Option", exact: true })`: `T = Option<X>` →
 *   no `undefined`, is `Option` → safe.
 * - `Schema.optional(X)` / `Schema.UndefinedOr(X)`: `T = X | undefined` →
 *   has `undefined`, not `Option` → **unsafe**.
 * - `Schema.NullOr(X)`: `T = X | null` → no `undefined` → safe.
 * - `optionalWith(X, { default: ... })`: `T = X` → no `undefined` → safe.
 *
 * We must check the decoded **struct Type** (not the field schema's own Type)
 * because `Schema.optional(X)` has field-schema `Type = X` (no `undefined`).
 * The `undefined` is introduced by the struct's property optionality.
 */
type IsFieldUnsafe<T, K extends keyof T> =
  [undefined] extends [T[K]]
    ? [T[K]] extends [Option.Option<infer _>] ? false : true
    : false

/**
 * `true` if ALL properties of `T` (indexed by keys from `F`) are safe.
 * Uses field-schema keys (`keyof F`) to index into the decoded type `T`,
 * because TaggedClass decoded types have `keyof T = string` (index
 * signature), which would prevent per-property checks.
 */
type AllPropertiesSafe<T, F> =
  true extends { readonly [K in keyof F]: K extends keyof T ? IsFieldUnsafe<T, K> : false }[keyof F]
    ? false
    : true

// ---------------------------------------------------------------------------
// Deep recursive safety check
// ---------------------------------------------------------------------------

/**
 * Recursively verify that a schema's decoded type contains no bare
 * `undefined` at any depth.
 *
 * Dispatch by schema structure:
 * 1. **Struct / TaggedClass** (has `.fields`): check each property of the
 *    decoded `Type` using `.fields` keys, then recurse into each field
 *    schema for nested structs/unions/arrays.
 * 2. **Union** (has `.members`): recurse into each member schema.
 * 3. **Array** (has `.value`): recurse into the element schema.
 * 4. **Transformation** (has `.from`): recurse into the inner schema.
 *    Covers `Schema.optional`, `Schema.UndefinedOr`, `Schema.NullOr`,
 *    `optionalWith`, refinements, and suspends.
 * 5. **Scalars / literals**: safe by default.
 *
 * **Why inspect schema structure, not just `Schema.Schema.Type`?**
 * TaggedClass decoded types have a `string` index signature, so
 * `keyof Type = string` and `AllPropertiesSafe` can't detect individual
 * optional fields. Schema objects expose `.fields` with proper literal key
 * types, which we use to index into the decoded `Type`.
 */
type IsSchemaSafe<S> =
  S extends { readonly fields: infer F }
    ? // Struct or TaggedClass — check each field via decoded Type
      (AllPropertiesSafe<Schema.Schema.Type<S>, F> extends true
        ? // All immediate fields safe — recurse into each field schema.
          // `false extends ...` detects if any field is unsafe (non-distributive).
          (false extends { readonly [K in keyof F]: IsSchemaSafe<F[K]> }[keyof F] ? false : true)
        : false)
    : S extends { readonly members: infer M }
      ? // Union — check each member
        (M extends readonly (infer E)[]
          ? (false extends { readonly [K in keyof M]: IsSchemaSafe<M[K]> }[number] ? false : true)
          : true)
      : S extends { readonly value: infer E }
        ? // Array — check element schema
          IsSchemaSafe<E>
        : S extends { readonly from: infer I }
          ? // Transformation (optional, NullOr, refinement, suspend, etc.)
            IsSchemaSafe<I>
          : // Scalar, literal, or unknown — safe
            true

// ---------------------------------------------------------------------------
// Public constraint
// ---------------------------------------------------------------------------

/**
 * Boolean check: `true` if `S`'s schema is JSON-safe, `false` otherwise.
 * Use this in conditional types that branch on safety.
 */
type IsSchemaJsonSafe<S extends Schema.Schema.AnyNoContext> = IsSchemaSafe<S> extends true ? true : false

/**
 * Compile-time constraint that rejects schemas whose decoded type contains
 * any property (at any depth) that is bare `undefined` instead of
 * `Option<T>`.
 *
 * Resolves to `S` for safe schemas and an error object for unsafe ones.
 * Used as a **field type constraint** in `compilePatchMap`.
 *
 * **Cannot** be used as `S extends JsonSafeSchema<S>` — TypeScript rejects
 * that as a circular type-parameter constraint (TS2313). The field-constraint
 * form is the correct usage.
 */
type JsonSafeSchema<S> = IsSchemaSafe<S> extends true ? S : never

/**
 * Intersection helper that surfaces a clear TS2345 error at the **call site**
 * when a schema is not JSON-safe.
 *
 * For safe schemas, resolves to just `C` (no extra fields).
 * For unsafe schemas, resolves to `C & { readonly __jsonSafeError: ErrorMsg }`,
 * so TypeScript reports the missing `__jsonSafeError` property on the object
 * literal passed by the caller — directly at the `define()` / `defineForked()`
 * call site, not inside library internals.
 */
type EnforceJsonSafe<S extends Schema.Schema.AnyNoContext, C> =
  (IsSchemaJsonSafe<S> extends true ? {} : {
    readonly __jsonSafeError: 'Schema contains bare Schema.optional() or Schema.UndefinedOr() — use optionalWith(X, { as: "Option", exact: true }) instead'
  }) & C

export type { JsonSafeSchema, EnforceJsonSafe, IsSchemaJsonSafe }
