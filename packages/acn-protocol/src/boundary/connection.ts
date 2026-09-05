import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import { MagnitudeHealthResponseSchema } from "../schemas/acn-health"

const Health = Rpc.make("Health", {
  payload: Schema.Struct({}),
  success: MagnitudeHealthResponseSchema,
}).pipe(replaySafe)

export const Connection = {
  health: Health,
}
