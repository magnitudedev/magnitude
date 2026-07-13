// This file is kept as a permanent proof-of-correctness test for JsonSafeSchema.
// Run: npx tsc --noEmit --moduleResolution bundler --module esnext --target es2022 --strict --skipLibCheck packages/utils/src/schema/__test_final.ts
//
// Expected output: exactly 7 errors, all on REJECT cases (lines marked ERROR).
// Zero errors on ACCEPT cases.

import { Schema, Option } from 'effect'
import type { JsonSafeSchema } from './json-safe'

// ---------------------------------------------------------------------------
// Test harness — field constraint pattern (same as compiled-map.ts and define.ts)
// ---------------------------------------------------------------------------

function compilePatchMap<S extends Schema.Schema.AnyNoContext>(schema: JsonSafeSchema<S>): void {}

interface ProjectionConfig<S extends Schema.Schema.AnyNoContext> {
  readonly state: JsonSafeSchema<S>
  readonly initial: Schema.Schema.Type<S>
}

function define<S extends Schema.Schema.AnyNoContext>(config: ProjectionConfig<S>): void {}

// ---------------------------------------------------------------------------
// ACCEPT cases (should compile with zero errors)
// ---------------------------------------------------------------------------

// 1. No optional fields
const NoOptional = Schema.Struct({ a: Schema.String, b: Schema.Number })
compilePatchMap(NoOptional)
define({ state: NoOptional, initial: { a: '', b: 0 } })

// 2. optionalWith as Option exact
const OptionExact = Schema.Struct({
  a: Schema.String,
  b: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
})
compilePatchMap(OptionExact)
define({ state: OptionExact, initial: { a: '', b: Option.none() } })

// 3. NullOr (null, not undefined)
const NullOrField = Schema.Struct({ a: Schema.String, b: Schema.NullOr(Schema.Number) })
compilePatchMap(NullOrField)

// 4. optionalWith default (Type is just X, not X|undefined)
const WithDefault = Schema.Struct({
  a: Schema.String,
  b: Schema.optionalWith(Schema.Number, { default: () => 0 }),
})
compilePatchMap(WithDefault)

// 5. Nested struct with Option
const NestedOption = Schema.Struct({
  a: Schema.String,
  inner: Schema.Struct({ x: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }) }),
})
compilePatchMap(NestedOption)

// 6. Empty struct
compilePatchMap(Schema.Struct({}))

// 7. Top-level scalars
compilePatchMap(Schema.String)
compilePatchMap(Schema.Number)

// 8. Array of safe structs
const SafeArray = Schema.Struct({
  items: Schema.Array(Schema.Struct({ x: Schema.String })),
})
compilePatchMap(SafeArray)

// 9. Union with all-safe members
const SafeUnion = Schema.Union(
  Schema.Struct({ type: Schema.Literal('a'), x: Schema.String }),
  Schema.Struct({ type: Schema.Literal('b'), y: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }) }),
)
compilePatchMap(SafeUnion)

// 10. optionalWith as Option (non-exact)
const NonExactOption = Schema.Struct({
  a: Schema.String,
  b: Schema.optionalWith(Schema.Number, { as: 'Option' }),
})
compilePatchMap(NonExactOption)

// ---------------------------------------------------------------------------
// REJECT cases (should produce compile errors)
// ---------------------------------------------------------------------------

// 11. Schema.optional (bare)
const BareOptional = Schema.Struct({ a: Schema.String, b: Schema.optional(Schema.Number) })
compilePatchMap(BareOptional) // ERROR

// 12. Schema.UndefinedOr
const UndefOr = Schema.Struct({ a: Schema.String, b: Schema.UndefinedOr(Schema.Number) })
compilePatchMap(UndefOr) // ERROR

// 13. Nested bare optional
const NestedBare = Schema.Struct({
  a: Schema.String,
  inner: Schema.Struct({ x: Schema.optional(Schema.String) }),
})
compilePatchMap(NestedBare) // ERROR

// 14. Array of unsafe structs
const UnsafeArray = Schema.Struct({
  items: Schema.Array(Schema.Struct({ x: Schema.optional(Schema.String) })),
})
compilePatchMap(UnsafeArray) // ERROR

// 15. Union with bare optional member
const UnsafeUnion = Schema.Union(
  Schema.Struct({ type: Schema.Literal('a'), x: Schema.String }),
  Schema.Struct({ type: Schema.Literal('b'), y: Schema.optional(Schema.String) }),
)
compilePatchMap(UnsafeUnion) // ERROR

// 16. define() with bare optional
define({ state: BareOptional, initial: { a: '' } }) // ERROR

// 17. Atif-like schema (mimicking production pattern)
const AtifLike = Schema.Struct({
  trajectory_id: Schema.optional(Schema.String),
  trajectory_path: Schema.optional(Schema.String),
  session_id: Schema.optional(Schema.String),
  extra: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.String })),
})
compilePatchMap(AtifLike) // ERROR
