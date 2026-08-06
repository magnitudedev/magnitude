import { Schema } from "effect"

export const ClientIdSchema = Schema.NonEmptyString.pipe(Schema.brand("ClientId"))
export type ClientId = typeof ClientIdSchema.Type

export const ConnectedClientCountSchema = Schema.Number.pipe(
  Schema.int(),
  Schema.nonNegative(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

export const ClientLeaseMutationResultSchema = Schema.Struct({
  connectedClientCount: ConnectedClientCountSchema,
})
export type ClientLeaseMutationResult = typeof ClientLeaseMutationResultSchema.Type
