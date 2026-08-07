import { Schema } from "effect"
import { AcnIdentitySchema } from "./acn-identity"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

export const AcnRevisionSchema = PositiveSafeInteger.pipe(
  Schema.brand("AcnRevision"),
)
export type AcnRevision = typeof AcnRevisionSchema.Type

export const AcnTargetSchema = Schema.Struct({
  revision: AcnRevisionSchema,
  identity: AcnIdentitySchema,
})
export type AcnTarget = typeof AcnTargetSchema.Type
