import * as Registry from "@effect-atom/atom/Registry"
import { Effect, Layer } from "effect"
import { describe, expect, it } from "vitest"
import { Client, Group, Mutation } from "@magnitudedev/effect-query"
import {
  PRIMARY_SLOT_ID,
  CatalogFormModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import { ModelSlots } from "../model-slots/service"
import { LocalModels } from "./service"
import { LocalModelOperations } from "./operations"

describe("SelectLocalModel", () => {
  it("runs synchronization before slot assignment through one tracked Mutation", async () => {
    const calls: string[] = []
    const modelId = CatalogFormModelIdSchema.make("selected-model:gguf:q4")
    const localModels = {
      install: (selected: typeof modelId) => Effect.sync(() => {
        calls.push(`sync:${selected}`)
      }),
    } as unknown as LocalModels
    const modelSlots = {
      assign: (_slotId: unknown, selection: { readonly providerModelId: string }) => Effect.sync(() => {
        calls.push(`assign:${selection.providerModelId}`)
      }),
    } as unknown as ModelSlots
    const services = Layer.merge(
      Layer.succeed(LocalModels, localModels),
      Layer.succeed(ModelSlots, modelSlots),
    )
    const operations = Group.make({ Models: LocalModelOperations })
    const client = Client.make(operations, Layer.empty, () => services)
    const registry = Registry.make()

    await Effect.runPromise(Mutation.execute(client.Models.SelectLocalModel, {
      modelId,
      slotId: PRIMARY_SLOT_ID,
      reasoningEffort: ReasoningEffortSchema.make("none"),
    }).pipe(Effect.provideService(Registry.AtomRegistry, registry)))

    expect(calls).toEqual([`sync:${modelId}`, `assign:${modelId}`])
    registry.dispose()
  })
})
