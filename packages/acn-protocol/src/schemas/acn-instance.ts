import { Schema } from "effect"
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  ProcessStartIdentitySchema,
} from "../acn-identity"
import { AcnHealthStateSchema } from "./acn-health"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

/** One exact ACN process as observed by a process manager. */
export const AcnInstanceSchema = Schema.Struct({
  id: AcnInstanceIdSchema,
  identity: AcnIdentitySchema,
  url: Schema.NonEmptyString,
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
  lifecycle: AcnHealthStateSchema,
})
export type AcnInstance = typeof AcnInstanceSchema.Type
