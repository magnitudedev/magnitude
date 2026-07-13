import { Schema } from 'effect'

const BadSchema = Schema.Struct({
  a: Schema.String,
  b: Schema.optional(Schema.String),
})

// Schema.Schema has .ast — what type is it?
type S = typeof BadSchema
type Ast = S['ast']
const _a: Ast = null as never

// Check if Schema.Schema.Type and Schema.Schema.Encoded are accessible
type Type = Schema.Schema.Type<S>
type Encoded = Schema.Schema.Encoded<S>
const _t: Type = null as never
const _e: Encoded = null as never

// The key: Schema.optional creates fields where Encoded includes undefined
// but Type does NOT include Option. optionalWith({ as: "Option" }) creates
// fields where Encoded includes undefined but Type IS Option<T>.
// So the check should be on the field types directly.
