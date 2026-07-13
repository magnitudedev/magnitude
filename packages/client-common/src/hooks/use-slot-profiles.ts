/**
 * Slot profiles hook — shared between web, desktop, and CLI.
 *
 * Derives slot profiles from the `GetCachedModelList` RPC response.
 */
import { useAtomValue, Result } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useDisplayState } from "../state/display-state-store"
import { ROLE_TO_SLOT, type SlotId } from "@magnitudedev/sdk"
import type { SlotProfile, SlotProfiles, ModelList } from "@magnitudedev/sdk"

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
}

export function useSlotProfiles(): UseSlotProfilesResult {
  const client = useAgentClient()

  const result = useAtomValue(
    client.query("GetCachedModelList", {}, { reactivityKeys: ["modelConfig"] }),
  )

  const profiles = Result.match(result, {
    onInitial: () => null,
    onFailure: () => null,
    onSuccess: (success) =>
      (success.value as ModelList).slotProfiles,
  })

  const rootRole = useDisplayState((state) => state.actors["root"]?.role ?? null)
  const rootRoleId = rootRole ?? "leader"
  const rootRoleLabel = formatRoleLabel(rootRoleId)
  const rootSlotId: SlotId = ROLE_TO_SLOT[rootRoleId as keyof typeof ROLE_TO_SLOT] ?? "primary"
  const rootProfile = findSlotProfile(profiles, rootSlotId)

  return { profiles, rootRoleId, rootRoleLabel, rootSlotId, rootProfile }
}
