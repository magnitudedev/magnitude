import { Schema } from "effect"
import { ProcessStartIdentitySchema } from "../acn-identity"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

export const ExactProcessSchema = Schema.Struct({
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
})
export type ExactProcess = typeof ExactProcessSchema.Type

export const AcnOwnerRecordSchema = Schema.Struct({
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
  port: PositiveSafeInteger.pipe(Schema.lessThanOrEqualTo(65_535)),
})
export type AcnOwnerRecord = typeof AcnOwnerRecordSchema.Type
