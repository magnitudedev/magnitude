import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { replaySafe } from "../transport/recovery"
import { InferenceObservationGroupIdSchema, InferenceObservationsSchema } from "../schemas/inference-observation"

export const Inference = {
  getObservations: Rpc.make("GetInferenceObservations", {
    payload: Schema.Struct({ groupId: InferenceObservationGroupIdSchema }),
    success: InferenceObservationsSchema,
  }).pipe(replaySafe),
}
