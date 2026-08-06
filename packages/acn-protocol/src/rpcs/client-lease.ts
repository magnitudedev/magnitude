import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import {
  ClientIdSchema,
  ClientLeaseMutationResultSchema,
} from "../schemas/client-lease"

const ClientLeaseMutationPayload = Schema.Struct({ clientId: ClientIdSchema })

export const RenewClientLease = Rpc.make("RenewClientLease", {
  payload: ClientLeaseMutationPayload,
  success: ClientLeaseMutationResultSchema,
})

export const ReleaseClientLease = Rpc.make("ReleaseClientLease", {
  payload: ClientLeaseMutationPayload,
  success: ClientLeaseMutationResultSchema,
})
