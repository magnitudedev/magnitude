/**
 * Overlays feature container (spec §5.6) — routes full-screen overlays:
 * recent chats, settings, usage, and worker fork detail. Visibility is pure
 * atom state; each overlay's data comes from shared hooks or display state.
 */
import { useCallback, useMemo, type ReactNode } from 'react'
import type { KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { Result, useAtomValue, useAtomSet } from '@effect-atom/atom-react'
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
  useLocalInferenceSnapshot,
} from '@magnitudedev/client-common'
import { forkIdToKey, ROLE_TO_SLOT, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS, isRoleId, type SlotId } from '@magnitudedev/sdk'
import { showRecentChatsOverlayAtom, authSourceAtom, modelSetupRouteAtom, type ModelSetupRoute } from '../../state/cli-atoms'
import type { ActionId } from '../../types/ui-actions'
import { deriveSettingsAuthInfo, type AuthInfo } from './auth-display'
import { SettingsOverlay } from './settings'
import { UsageOverlay } from './usage'
import { ForkDetailOverlay } from './fork-detail'
import { RecentChatsOverlayContainer } from '../sessions/container'
import { ModelSetupOnboardingScreen, PreparingLocalInferenceScreen } from '../local-inference-onboarding'

export type ActiveOverlay = 'recent-chats' | 'model-setup' | 'settings' | 'usage' | 'fork' | 'none'

export function useActiveOverlay(): ActiveOverlay {
  const showRecentChats = useAtomValue(showRecentChatsOverlayAtom)
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const modelSetupRoute = useAtomValue(modelSetupRouteAtom)
  const usageOpen = useAtomValue(usageOpenAtom)
  const { expandedForkStack } = useDisplayViewController()
  return showRecentChats ? 'recent-chats'
    : modelSetupRoute !== 'closed' ? 'model-setup'
    : settingsOpen ? 'settings'
    : usageOpen ? 'usage'
    : expandedForkStack.length > 0 ? 'fork'
    : 'none'
}

function ModelSetupSettingsContainer({
  route,
  onClose,
}: {
  readonly route: Exclude<ModelSetupRoute, 'closed'>
  readonly onClose: () => void
}): ReactNode {
  const snapshot = useLocalInferenceSnapshot()
  const childHandlesKeyboard = Result.isSuccess(snapshot)
  useKeyboard(useCallback((key: KeyEvent) => {
    if (!childHandlesKeyboard && key.name === 'escape') {
      key.preventDefault()
      onClose()
    }
  }, [childHandlesKeyboard, onClose]))
  if (Result.isInitial(snapshot)) return <PreparingLocalInferenceScreen />
  if (Result.isFailure(snapshot)) {
    return (
      <box style={{ height: '100%', alignItems: 'center', justifyContent: 'center' }}>
        <text>Failed to inspect local inference capabilities. Press Esc to return to Settings.</text>
      </box>
    )
  }
  return (
    <ModelSetupOnboardingScreen
      key={route}
      snapshot={snapshot.value}
      initialStep={route}
      mode="management"
      onExit={onClose}
      onComplete={onClose}
    />
  )
}

export function AppOverlaysContainer({
  dispatchErrorAction,
}: {
  dispatchErrorAction: (actionId: ActionId) => void
}): ReactNode {
  const active = useActiveOverlay()
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const modelSetupRoute = useAtomValue(modelSetupRouteAtom)
  const setModelSetupRoute = useAtomSet(modelSetupRouteAtom)
  const controller = useDisplayViewController()
  const displayMode = controller.displayMode
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const expandedForkId = controller.topForkId

  const { apiKey, saveApiKey, disconnectApiKey } = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const { profiles } = useSlotProfiles()
  const modelConfig = useModelConfig()
  const { pushFork, popFork } = controller

  const auth: AuthInfo = useMemo(() => deriveSettingsAuthInfo({
    apiKey,
    authSource,
    save: saveApiKey,
    clear: disconnectApiKey,
  }), [apiKey, authSource, saveApiKey, disconnectApiKey])

  const slots = useMemo(() => {
    return SLOT_IDS.map((slotId) => ({
      slotId,
      label: SLOT_DISPLAY_NAMES[slotId],
      description: SLOT_DESCRIPTIONS[slotId],
      modelDisplayName: profiles?.[slotId]?.modelDisplayName ?? null,
    }))
  }, [profiles])

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

  if (active === 'model-setup' && modelSetupRoute !== 'closed') {
    return (
      <ModelSetupSettingsContainer
        route={modelSetupRoute}
        onClose={() => setModelSetupRoute('closed')}
      />
    )
  }

  if (active === 'settings') {
    return (
      <SettingsOverlay
        isVisible
        onClose={() => setSettingsOpen(false)}
        auth={auth}
        slots={slots}
        modelConfig={modelConfig}
        onManageLocalModels={() => setModelSetupRoute('local')}
        onConfigureCloud={() => setModelSetupRoute('cloud')}
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
