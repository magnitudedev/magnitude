import { Schema } from "effect"
export const ProcessStartIdentitySchema = Schema.NonEmptyString.pipe(Schema.brand("ProcessStartIdentity"))
export type ProcessStartIdentity = typeof ProcessStartIdentitySchema.Type

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

export const ProcessGroupSchema = Schema.Struct({
  leader: ExactProcessSchema,
})
export type ProcessGroup = typeof ProcessGroupSchema.Type
