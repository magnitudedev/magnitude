/**
 * Overlays feature container (spec §5.6) — routes the remaining full-screen
 * overlays: recent chats, usage, and worker fork detail. Model management is
 * owned by the bottom-docked model-menu shell.
 */
import { useMemo, type ReactNode } from 'react'
import { useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import {
  useDisplayState,
  useSlotProfiles,
  getFork,
  usageOpenAtom,
  useDisplayViewController,
  selectedCwdAtom,
  useTimelineStatus,
  findSlotProfile,
} from '@magnitudedev/client-common'
import { forkIdToKey, PRIMARY_SLOT_ID, ROLE_TO_SLOT, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS, isRoleId, SECONDARY_SLOT_ID, type SlotId } from '@magnitudedev/sdk'
import { Option } from 'effect'
import { showRecentChatsOverlayAtom } from '../../state/cli-atoms'
import type { ActionId } from '../../types/ui-actions'
import { UsageOverlay } from './usage'
import { ForkDetailOverlay } from './fork-detail'
import { RecentChatsOverlayContainer } from '../sessions/container'

export type ActiveOverlay = 'recent-chats' | 'usage' | 'fork' | 'none'

export function useActiveOverlay(): ActiveOverlay {
  const showRecentChats = useAtomValue(showRecentChatsOverlayAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const { expandedForkStack } = useDisplayViewController()
  return showRecentChats ? 'recent-chats'
    : usageOpen ? 'usage'
    : expandedForkStack.length > 0 ? 'fork'
    : 'none'
}

export function AppOverlaysContainer({
  dispatchErrorAction,
}: {
  dispatchErrorAction: (actionId: ActionId) => void
}): ReactNode {
  const active = useActiveOverlay()
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const controller = useDisplayViewController()
  const displayMode = controller.displayMode
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const expandedForkId = controller.topForkId

  const { profiles } = useSlotProfiles()
  const { pushFork, popFork } = controller

  const forkActor = useDisplayState((state) =>
    expandedForkId ? state.actors[forkIdToKey(expandedForkId)] ?? null : null
  )
  const forkTimeline = useDisplayState((state) =>
    (expandedForkId ? getFork(state, expandedForkId) : null) ?? null
  )
  const forkTimelineStatus = useTimelineStatus(expandedForkId)

  if (active === 'none') return null

  if (active === 'recent-chats') {
    return <RecentChatsOverlayContainer />
  }

  if (active === 'usage') {
    return <UsageOverlay isVisible onClose={() => setUsageOpen(false)} />
  }

  // Fork detail
  const forkSlotName = forkActor && isRoleId(forkActor.role) ? ROLE_TO_SLOT[forkActor.role] : null
  const forkSlot: SlotId | null = forkSlotName === null
    ? null
    : forkSlotName === 'primary' ? PRIMARY_SLOT_ID : SECONDARY_SLOT_ID
  const forkProfile = forkSlot && profiles
    ? Option.getOrNull(findSlotProfile(profiles, forkSlot))
    : null
  const roleLabel = forkActor?.role ? forkActor.role.charAt(0).toUpperCase() + forkActor.role.slice(1) : ''
  const modelSummary = forkSlot
    ? { role: roleLabel, model: forkProfile?.modelDisplayName ?? '-' }
    : null

  return (
    <ForkDetailOverlay
      forkName={forkActor?.name ?? expandedForkId ?? ''}
      forkRole={forkActor?.role ?? ''}
      timeline={forkTimeline}
      timelineStatus={forkTimelineStatus._tag}
      context={forkActor?.context ?? null}
      displayMode={displayMode}
      onClose={popFork}
      onForkExpand={pushFork}
      onErrorAction={dispatchErrorAction}
      modelSummary={modelSummary}
      contextHardCap={forkProfile?.contextWindow ?? null}
      cwd={selectedCwd}
      projectRoot={process.cwd()}
    />
  )
}
