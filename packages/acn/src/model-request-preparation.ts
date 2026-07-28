import { Effect } from "effect"
import {
  ModelRequestPreparationFailed,
  type PrepareModelRequest,
} from "@magnitudedev/agent"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import type { ModelSlotControllerApi } from "./model-slot-controller"

export const makeModelRequestPreparation = (
  modelSlots: Pick<ModelSlotControllerApi, "acquireLocalModel">,
): PrepareModelRequest => ({ slotId, providerId, providerModelId }) => {
  if (providerId !== LOCAL_PROVIDER_ID) return Effect.void

  return modelSlots.acquireLocalModel(slotId, providerModelId).pipe(
    Effect.mapError((cause) => new ModelRequestPreparationFailed({
      code: "code" in cause ? cause.code : cause._tag,
      message: cause.message,
      retryable: "retryable" in cause ? cause.retryable : false,
    })),
  )
}
