import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import {
  isRoleId,
  Configuration,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  ROLE_TO_SLOT,
  SECONDARY_SLOT_ID,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
  type ReasoningEffort,
  type SlotId,
} from "@magnitudedev/sdk"
import { useDisplayState } from "../state/display-state-store"
import { useAgentClient } from "../state/agent-client-context"
import { ModelSlots } from "../model-slots/service"
import { formatModelDisplayName } from "../utils/model-presentation"

export interface SlotProfile {
  readonly slotId: SlotId
  readonly providerId: ProviderModelCatalogEntry["providerId"]
  readonly providerModelId: ProviderModelCatalogEntry["providerModelId"]
  readonly modelDisplayName: string
  readonly contextWindow: number
  readonly maxOutputTokens: number
  readonly reasoningEffort: ReasoningEffort
}

export interface SlotProfiles {
  readonly primary?: SlotProfile
  readonly secondary?: SlotProfile
}

export const findSlotProfile = (
  profiles: SlotProfiles,
  slotId: SlotId,
): Option.Option<SlotProfile> => Option.fromNullable(
  slotId === PRIMARY_SLOT_ID ? profiles.primary : profiles.secondary,
)

const catalogModels = (state: ProviderModelCatalogState): readonly ProviderModelCatalogEntry[] =>
  ProviderModelCatalogLifecycle.match(state, {
    Loading: () => [],
    Ready: ({ models }) => models,
    Refreshing: ({ models }) => models,
    Degraded: ({ models }) => models,
    Unavailable: () => [],
  })

export function useSlotProfiles() {
  const client = useAgentClient()
  const slotService = useMemo(() => client.runtime.atom(ModelSlots), [client])
  const slotState = useMemo(() => Atom.make((get) => Result.flatMap(
    get(slotService),
    (service) => get(service.state),
  )), [slotService])
  const slots = useAtomValue(slotState)
  const catalogAtom = useMemo(() => Atom.make((get) => get(client.query(Configuration.GetProviderModelCatalog, {})).result), [client])
  const catalog = useAtomValue(catalogAtom)
  const refreshAtom = useMemo(() => client.mutation(Configuration.RefreshModelCatalog), [client])
  const refresh = useAtomSet(refreshAtom)
  const retry = useCallback(() => refresh({ providerId: Option.none() }), [refresh])

  const profiles = Option.flatMap(Result.value(slots), ({ state: slotState }) =>
    Option.map(Result.value(catalog), ({ state: catalogState }): SlotProfiles => {
      const models = catalogModels(catalogState)
      const profile = (slot: typeof slotState.slots.primary): SlotProfile | undefined => {
        if (slot._tag === "Unassigned") return undefined
        const model = models.find((candidate) => candidate.providerId === slot.selection.providerId
          && candidate.providerModelId === slot.selection.providerModelId)
        if (!model) return undefined
        return {
          slotId: slot.slotId,
          providerId: slot.selection.providerId,
          providerModelId: slot.selection.providerModelId,
          modelDisplayName: formatModelDisplayName(model.displayName, model.variantLabel),
          contextWindow: model.contextWindow,
          maxOutputTokens: model.maxOutputTokens,
          reasoningEffort: slot.selection.reasoningEffort,
        }
      }
      const primary = profile(slotState.slots.primary)
      const secondary = profile(slotState.slots.secondary)
      return {
        ...(primary ? { primary } : {}),
        ...(secondary ? { secondary } : {}),
      }
    }))

  const rootRole = useDisplayState((state) => state.actors["root"]?.role ?? null)
  const rootRoleId = rootRole ?? "leader"
  const rootRoleLabel = rootRoleId.charAt(0).toUpperCase() + rootRoleId.slice(1)
  const roleSlot = isRoleId(rootRoleId) ? ROLE_TO_SLOT[rootRoleId] : "primary"
  const rootSlotId = roleSlot === "primary" ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  const values = Option.getOrNull(profiles)
  return {
    profiles: values,
    rootRoleId,
    rootRoleLabel,
    rootSlotId,
    rootProfile: Option.getOrNull(Option.flatMap(profiles, (slotProfiles) =>
      findSlotProfile(slotProfiles, rootSlotId))),
    slots,
    retry,
  }
}
