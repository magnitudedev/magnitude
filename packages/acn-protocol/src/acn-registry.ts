import { Schema } from "effect"

export const AcnOwnerIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.brand("AcnOwnerId")
)
export type AcnOwnerId = typeof AcnOwnerIdSchema.Type

export const AcnRegistrationSchema = Schema.Struct({
  id: AcnOwnerIdSchema,
  version: Schema.String,
  url: Schema.String,
  pid: Schema.Number,
  timestamp: Schema.Number,
})
export type AcnRegistration = Schema.Schema.Type<typeof AcnRegistrationSchema>

/**
 * Stable cross-version projection used only to determine whether an ACN
 * remains the canonical process. Additional registration fields are ignored.
 */
export const AcnRegistrationOwnershipSchema = Schema.Struct({
  id: AcnOwnerIdSchema,
})
export type AcnRegistrationOwnership =
  Schema.Schema.Type<typeof AcnRegistrationOwnershipSchema>

export const AcnVersionRegistrySchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  registration: Schema.optionalWith(AcnRegistrationSchema, {
    as: "Option",
    exact: true,
  }),
})
export type AcnVersionRegistry = Schema.Schema.Type<typeof AcnVersionRegistrySchema>

export const AcnVersionRegistryOwnershipSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  registration: Schema.optionalWith(AcnRegistrationOwnershipSchema, {
    as: "Option",
    exact: true,
  }),
})
export type AcnVersionRegistryOwnership =
  Schema.Schema.Type<typeof AcnVersionRegistryOwnershipSchema>

export const AcnVersionRegistryJson = Schema.parseJson(AcnVersionRegistrySchema)
