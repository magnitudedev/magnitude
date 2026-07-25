import { Effect } from "effect"
import {
  ModelRequestPreparationFailed,
  type PrepareModelRequest,
} from "@magnitudedev/agent"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import type { ModelSlotCoordinatorApi } from "./model-slot-coordinator"

export const makeModelRequestPreparation = (
  modelSlots: Pick<ModelSlotCoordinatorApi, "acquireLocalModel">,
): PrepareModelRequest => ({ slotId, providerId, providerModelId, reportProgress }) => {
  if (providerId !== LOCAL_PROVIDER_ID) return Effect.void

  return reportProgress({
    phase: "preparing",
    requestId: null,
  }).pipe(
    Effect.zipRight(modelSlots.acquireLocalModel(slotId, providerModelId)),
    Effect.mapError((cause) => new ModelRequestPreparationFailed({
      code: "code" in cause ? cause.code : cause._tag,
      message: cause.message,
      retryable: "retryable" in cause ? cause.retryable : false,
    })),
  )
}
