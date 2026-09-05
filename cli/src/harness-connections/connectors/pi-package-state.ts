import { Schema } from "effect"

export const PiPackageSourceSchema = Schema.NonEmptyString.pipe(Schema.brand("PiPackageSource"))
export type PiPackageSource = typeof PiPackageSourceSchema.Type

const fields = { identity: Schema.Literal("@magnitudedev/pi-extension"), source: PiPackageSourceSchema }
/** A receipt records only the package and fields Magnitude is entitled to undo. */
export const PiCompanionStateSchema = Schema.Union(
  Schema.Struct({ ...fields, ownership: Schema.Literal("magnitude") }),
  Schema.Struct({
    ...fields,
    ownership: Schema.Literal("pre-existing"),
    enablement: Schema.optionalWith(Schema.Struct({
      before: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
      after: Schema.Array(Schema.String),
    }), { as: "Option", exact: true }),
  }),
)
