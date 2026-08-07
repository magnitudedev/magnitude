import { Schema } from "effect"
import {
  AcnHealthStateSchema,
  AcnReady,
  type AcnHealthState,
} from "./schemas/acn-health"
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  ProcessStartIdentitySchema,
} from "./acn-identity"
import { AcnRevisionSchema, type AcnRevision } from "./acn-revision"

const PositiveSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

/** Fields shared by the general instance schema and every lifecycle refinement. */
export const AcnInstanceFields = {
  revision: AcnRevisionSchema,
  id: AcnInstanceIdSchema,
  identity: AcnIdentitySchema,
  url: Schema.NonEmptyString,
  pid: PositiveSafeInteger,
  processStartIdentity: ProcessStartIdentitySchema,
} as const

export interface AcnInstance<State extends AcnHealthState = AcnHealthState> {
  readonly revision: AcnRevision
  readonly id: typeof AcnInstanceIdSchema.Type
  readonly identity: typeof AcnIdentitySchema.Type
  readonly url: string
  readonly pid: number
  readonly processStartIdentity: typeof ProcessStartIdentitySchema.Type
  readonly lifecycle: State
}

export const AcnInstanceSchema = Schema.Struct({
  ...AcnInstanceFields,
  lifecycle: AcnHealthStateSchema,
})

export const AcnReadyInstanceSchema = Schema.Struct({
  ...AcnInstanceFields,
  lifecycle: AcnReady,
})
