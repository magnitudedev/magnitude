import { Effect } from "effect"
import { Group, Mutation } from "@magnitudedev/effect-query"
import {
  ProviderIdSchema,
  type CatalogFormModelId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { ModelSlots } from "../model-slots/service"
import { LocalModels } from "./service"

export interface SelectLocalModelInput {
  readonly modelId: CatalogFormModelId
  readonly slotId: SlotId
  readonly reasoningEffort: SlotSelection["reasoningEffort"]
}

/** Model-menu orchestration: synchronize the selected model, then assign it to the chosen slot. */
export const SelectLocalModel = Mutation.make("SelectLocalModel", {
  scope: ({ slotId }: SelectLocalModelInput) => Mutation.MutationScope(`model-selection:${slotId}`),
  effect: ({ modelId, slotId, reasoningEffort }: SelectLocalModelInput) => Effect.gen(function* () {
    const models = yield* LocalModels
    const slots = yield* ModelSlots

    yield* models.install(modelId)
    yield* slots.assign(slotId, {
      providerId: ProviderIdSchema.make("local"),
      providerModelId: modelId,
      reasoningEffort,
    })
  }),
})

export const LocalModelOperations = Group.make({ SelectLocalModel })
