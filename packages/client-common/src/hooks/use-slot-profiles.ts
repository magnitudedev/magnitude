/**
 * Slot profiles hook — shared between web, desktop, and CLI.
 *
 * Reads the authoritative reactive model-slot resource.
 */
import { Result } from "@effect-atom/atom-react"
import { useDisplayState } from "../state/display-state-store"
import { isRoleId, ROLE_TO_SLOT, type SlotId } from "@magnitudedev/sdk"
import type { SlotProfile, SlotProfiles } from "@magnitudedev/sdk"
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
  /** All slot profiles (null while loading/failed) */
  profiles: Partial<Record<SlotId, SlotProfile>> | null
  /** Root agent's role id */
  rootRoleId: string
  /** Root agent's role label (capitalized) */
  rootRoleLabel: string
  /** Root agent's slot id */
  rootSlotId: SlotId
  /** Root agent's slot profile (null if not found) */
  rootProfile: SlotProfile | null
  /** Whether the authoritative model configuration is still loading. */
  loading: boolean
}

export function useSlotProfiles(): UseSlotProfilesResult {
  const result = useModelSlots()

  const profiles = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) => success.value.profiles,
  })

  const rootRole = useDisplayState((state) => state.actors["root"]?.role ?? null)
  const rootRoleId = rootRole ?? "leader"
  const rootRoleLabel = formatRoleLabel(rootRoleId)
  const rootSlotId: SlotId = isRoleId(rootRoleId) ? ROLE_TO_SLOT[rootRoleId] : "primary"
  const rootProfile = findSlotProfile(profiles, rootSlotId)

  return {
    profiles,
    rootRoleId,
    rootRoleLabel,
    rootSlotId,
    rootProfile,
    loading: Result.isInitial(result),
  }
}
