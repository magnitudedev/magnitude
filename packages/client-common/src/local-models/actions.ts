import { Atom } from "@effect-atom/atom-react"
import { Effect } from "effect"
import { Mutation } from "@magnitudedev/effect-query"
import {
  ProviderIdSchema,
  type ModelServingConfigurationId,
  type SlotId,
  type SlotSelection,
} from "@magnitudedev/sdk"
import type { AgentClientInstance } from "../state/agent-client"
import { modelSlotAtoms } from "../model-slots/atoms"
import { localModelAtoms } from "./atoms"

export interface InstallAndAssignLocalModelInput {
  readonly configurationId: ModelServingConfigurationId
  readonly slotId: SlotId
  readonly reasoningEffort: SlotSelection["reasoningEffort"]
}

const makeAtoms = (client: AgentClientInstance) => {
  const localModels = localModelAtoms(client)
  const slots = modelSlotAtoms(client)
  const installAndAssign = Atom.keepAlive(client.effectQuery.runtime.fn<InstallAndAssignLocalModelInput>()(
    ({ configurationId, slotId, reasoningEffort }) => Mutation.execute(
      localModels.installMutation,
      { configurationId },
    ).pipe(
      Effect.flatMap(({ providerModelId }) => Mutation.execute(slots.assignMutation, {
        slotId,
        selection: {
          providerId: ProviderIdSchema.make("local"),
          providerModelId,
          reasoningEffort,
        },
      })),
    ),
  ))
  return { installAndAssign }
}

export type LocalModelActionAtoms = ReturnType<typeof makeAtoms>

const atomsByClient = new WeakMap<object, LocalModelActionAtoms>()

export const localModelActionAtoms = (client: AgentClientInstance): LocalModelActionAtoms => {
  const existing = atomsByClient.get(client)
  if (existing !== undefined) return existing
  const atoms = makeAtoms(client)
  atomsByClient.set(client, atoms)
  return atoms
}
