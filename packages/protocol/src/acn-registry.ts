import { Schema } from "effect"

export const AcnRegistrationSchema = Schema.Struct({
  id: Schema.String,
  version: Schema.String,
  url: Schema.String,
  pid: Schema.Number,
  timestamp: Schema.Number
})
export type AcnRegistration = Schema.Schema.Type<typeof AcnRegistrationSchema>

export const AcnVersionRegistrySchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  registration: Schema.Union(AcnRegistrationSchema, Schema.Null)
})
export type AcnVersionRegistry = Schema.Schema.Type<typeof AcnVersionRegistrySchema>

export const AcnVersionRegistryJson = Schema.parseJson(AcnVersionRegistrySchema)

export function versionPathSegment(version: string): string {
  return encodeURIComponent(version)
}
