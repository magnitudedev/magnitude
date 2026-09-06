import { Schema } from "effect"

export const HermesPackageSourceSchema = Schema.NonEmptyString.pipe(Schema.brand("HermesPackageSource"))
export const HermesPackageRevisionSchema = Schema.String.pipe(
  Schema.pattern(/^[a-f0-9]{40}$/), Schema.brand("HermesPackageRevision"),
)
export const HermesPackageSelectionSchema = Schema.Struct({
  source: HermesPackageSourceSchema,
  revision: HermesPackageRevisionSchema,
  contentFingerprint: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/)),
})
export type HermesPackageSelection = typeof HermesPackageSelectionSchema.Type

const fields = {
  identity: Schema.Literal("@magnitudedev/hermes-companion"),
  enablement: Schema.optionalWith(Schema.Struct({
    enabled: Schema.Boolean,
    disabled: Schema.Boolean,
  }), { as: "Option", exact: true }),
}

/** Only a verified native installation made by Magnitude can be removed by it. */
export const HermesCompanionStateSchema = Schema.Union(
  Schema.Struct({ ...fields, ...HermesPackageSelectionSchema.fields, ownership: Schema.Literal("magnitude") }),
  Schema.Struct({ ...fields, source: HermesPackageSourceSchema, ownership: Schema.Literal("pre-existing") }),
)
