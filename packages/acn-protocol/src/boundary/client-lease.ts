import { Schema } from "effect"
import { Group, Mutation } from "@magnitudedev/effect-query"
import {
  ClientIdSchema,
  ClientLeaseMutationResultSchema,
} from "../schemas/client-lease"

const ClientLeaseMutationPayload = Schema.Struct({ clientId: ClientIdSchema })

/** Lease renewal; executed by the transport's JIT runtime, never by client code. */
const RenewClientLease = Mutation.make("RenewClientLease", {
  policy: { recovery: "ReplaySafe", demand: false },
  payload: ClientLeaseMutationPayload,
  success: ClientLeaseMutationResultSchema,
})

/** Graceful lease release; executed by the transport's JIT runtime on close. */
const ReleaseClientLease = Mutation.make("ReleaseClientLease", {
  policy: { recovery: "ReplaySafe", demand: false },
  payload: ClientLeaseMutationPayload,
  success: ClientLeaseMutationResultSchema,
})

export const ClientLease = Group.make({ RenewClientLease, ReleaseClientLease })
