/**
 * Overlays feature container (spec §5.6) — routes full-screen overlays:
 * recent chats, settings, usage, and worker fork detail. Visibility is pure
 * atom state; each overlay's data comes from shared hooks or display state.
 */
import type { ReactNode } from 'react'
import { useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import {
  useDisplayState,
  useSettingsState,
  useSlotProfiles,
  useModelConfig,
  getFork,
  settingsOpenAtom,
  usageOpenAtom,
  useDisplayViewController,
  selectedCwdAtom,
  useTimelineStatus,
} from '@magnitudedev/client-common'
import { forkIdToKey, ROLE_TO_SLOT, isRoleId, type SlotId } from '@magnitudedev/sdk'
import { showRecentChatsOverlayAtom } from '../../state/cli-atoms'
import type { ActionId } from '../../types/ui-actions'
import { SettingsOverlay } from './settings'
import { UsageOverlay } from './usage'
import { ForkDetailOverlay } from './fork-detail'
import { RecentChatsOverlayContainer } from '../sessions/container'

export type ActiveOverlay = 'recent-chats' | 'settings' | 'usage' | 'fork' | 'none'

export function useActiveOverlay(): ActiveOverlay {
  const showRecentChats = useAtomValue(showRecentChatsOverlayAtom)
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const { expandedForkStack } = useDisplayViewController()
  return showRecentChats ? 'recent-chats'
    : settingsOpen ? 'settings'
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
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const controller = useDisplayViewController()
  const displayMode = controller.displayMode
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const expandedForkId = controller.topForkId

  const settings = useSettingsState()
  const { profiles } = useSlotProfiles()
  const modelConfig = useModelConfig()
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

  if (active === 'settings') {
    return (
      <SettingsOverlay
        isVisible
        onClose={() => setSettingsOpen(false)}
        providerAuths={settings.providerAuths}
        onSaveProviderApiKey={settings.saveProviderApiKey}
        onDisconnectProvider={settings.disconnectProvider}
        modelConfig={modelConfig}
      />
    )
  }

  if (active === 'usage') {
    return <UsageOverlay isVisible onClose={() => setUsageOpen(false)} />
  }

  // Fork detail
  const forkSlot: SlotId | null = forkActor && isRoleId(forkActor.role) ? ROLE_TO_SLOT[forkActor.role] : null
  const roleLabel = forkActor?.role ? forkActor.role.charAt(0).toUpperCase() + forkActor.role.slice(1) : ''
  const modelSummary = forkSlot
    ? { role: roleLabel, model: profiles?.[forkSlot]?.modelDisplayName ?? '-' }
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
      contextHardCap={forkSlot ? profiles?.[forkSlot]?.contextWindow ?? null : null}
      cwd={selectedCwd}
      projectRoot={process.cwd()}
    />
  )
}
