/**
 * Slot profiles hook — shared between web, desktop, and CLI.
 *
 * Reads the authoritative reactive model-slot resource.
 */
import { useCallback, useMemo } from "react"
import { Result, useAtomSet } from "@effect-atom/atom-react"
import { Option } from "effect"
import { useDisplayState } from "../state/display-state-store"
import { isRoleId, ModelSlotsLifecycle, ROLE_TO_SLOT, type SlotId } from "@magnitudedev/sdk"
import type { SlotProfile, SlotProfiles } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useModelSlots } from "./use-reactive-rpc"

/**
 * Find the slot profile for a given slot ID.
 */
export function findSlotProfile(
  profiles: SlotProfiles | null,
  slotId: SlotId | null | undefined,
): SlotProfile | null {
  if (!profiles || !slotId) return null
  return profiles[slotId] ?? null
}

function formatRoleLabel(role: string | null | undefined): string {
  if (!role) return "Leader"
  return role.charAt(0).toUpperCase() + role.slice(1)
}

export interface UseSlotProfilesResult {
  /** Best available profiles selected from the authoritative state; inspect `slots` for lifecycle meaning. */
  profiles: Partial<Record<SlotId, SlotProfile>> | null
  /** Root agent's role id */
  rootRoleId: string
  /** Root agent's role label (capitalized) */
  rootRoleLabel: string
  /** Root agent's slot id */
  rootSlotId: SlotId
  /** Root agent's slot profile (null if not found) */
  rootProfile: SlotProfile | null
  /** Transport result containing the authoritative versioned FSM state. */
  slots: ReturnType<typeof useModelSlots>
  /** Retry authoritative model discovery after a terminal load failure. */
  retry: () => void
}

export function useSlotProfiles(): UseSlotProfilesResult {
  const client = useAgentClient()
  const result = useModelSlots()
  const refreshAtom = useMemo(() => client.mutation("RefreshModelCatalog"), [client])
  const refresh = useAtomSet(refreshAtom)
  const retry = useCallback(() => {
    refresh({ payload: { providerId: Option.none() }, reactivityKeys: ["modelCatalog", "modelSlots"] })
  }, [refresh])

  const profiles = Option.flatMap(Result.value(result), ({ state }) => ModelSlotsLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ profiles }) => Option.some(profiles),
    refreshing: ({ profiles }) => Option.some(profiles),
    degraded: ({ profiles }) => Option.some(profiles),
    unavailable: () => Option.none(),
  }))

  const rootRole = useDisplayState((state) => state.actors["root"]?.role ?? null)
  const rootRoleId = rootRole ?? "leader"
  const rootRoleLabel = formatRoleLabel(rootRoleId)
  const rootSlotId: SlotId = isRoleId(rootRoleId) ? ROLE_TO_SLOT[rootRoleId] : "primary"
  const rootProfile = Option.match(profiles, {
    onNone: () => null,
    onSome: (value) => findSlotProfile(value, rootSlotId),
  })

  return {
    profiles: Option.getOrNull(profiles),
    rootRoleId,
    rootRoleLabel,
    rootSlotId,
    rootProfile,
    slots: result,
    retry,
  }
}
