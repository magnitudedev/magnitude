import { memo, useCallback, useState, useMemo } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { Atom, Result, useAtomMount } from '@effect-atom/atom-react'
import { Effect, Option } from 'effect'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { SingleLineInput } from '../composer/single-line-input'
import type { AuthInfo } from './auth-display'
import type { UseModelConfigResult } from '@magnitudedev/client-common'
import { ModelCatalogLifecycle, ModelSlotsLifecycle, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS, DEFAULT_REASONING_EFFORT, type ProviderCatalogFailure, type SlotId } from '@magnitudedev/sdk'
import { getInferenceSourceAction, INFERENCE_SOURCE_ACTIONS } from './inference-source-actions'

interface SettingsOverlayProps {
  isVisible: boolean
  onClose: () => void
  auth: AuthInfo
  slots: ReadonlyArray<{
    slotId: SlotId
    label: string
    description: string
    modelDisplayName: string | null
  }>
  modelConfig?: UseModelConfigResult
  onManageLocalModels: () => void
  onConfigureCloud: () => void
}

type Mode = 'view' | 'edit' | 'confirm-disconnect'
type PendingAuthAction = 'save' | 'clear' | null

type DropdownTarget =
  | { slotId: SlotId; field: 'model' }
  | { slotId: SlotId; field: 'thinking' }
  | null

const REASONING_OPTIONS: { value: string; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'max', label: 'Max' },
]

function formatContextWindow(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

function formatPricing(pricing: { input: number; output: number; cachedInput?: number }): string {
  return `$${pricing.input.toFixed(2)}/$${pricing.output.toFixed(2)}`
}

const formatCatalogFailures = (failures: readonly ProviderCatalogFailure[]): string => failures.length === 1
  ? `${failures[0]!.providerId}: ${failures[0]!.message}`
  : `${failures.length} model providers failed.`

const disabledReasonLabel = (reason: "insufficient_resources" | "provider_unavailable" | "model_unavailable" | "incompatible_runtime" | "invalid_configuration"): string => ({
  insufficient_resources: 'not enough free memory',
  provider_unavailable: 'server unavailable',
  model_unavailable: 'model unavailable',
  incompatible_runtime: 'incompatible runtime',
  invalid_configuration: 'invalid configuration',
})[reason]

function maskApiKey(key: string): string {
  const trimmed = key.trim()
  if (trimmed.length <= 12) return '•'.repeat(Math.max(trimmed.length, 4))
  const lastUnderscore = trimmed.lastIndexOf('_')
  const head = lastUnderscore >= 0 && lastUnderscore < trimmed.length - 8
    ? trimmed.slice(0, lastUnderscore + 1 + 4)
    : trimmed.slice(0, 6)
  const tail = trimmed.slice(-4)
  return `${head}………${tail}`
}

export const SettingsOverlay = memo(function SettingsOverlay({
  isVisible,
  onClose,
  auth,
  slots,
  modelConfig,
  onManageLocalModels,
  onConfigureCloud,
}: SettingsOverlayProps) {
  const theme = useTheme()
  const [mode, setMode] = useState<Mode>('view')
  const [inputValue, setInputValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [pendingAuthAction, setPendingAuthAction] = useState<PendingAuthAction>(null)
  const displayedAuthError = error ?? auth.error

  const catalogSnapshot = modelConfig ? Result.value(modelConfig.catalog) : Option.none()
  const catalogState = Option.map(catalogSnapshot, ({ state }) => state)
  const models = Option.getOrNull(Option.flatMap(catalogState, (state) => ModelCatalogLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ models }) => Option.some(models),
    refreshing: ({ models }) => Option.some(models),
    degraded: ({ models }) => Option.some(models),
    unavailable: () => Option.none(),
  })))
  const providers = Option.getOrNull(Option.flatMap(catalogState, (state) => ModelCatalogLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ providers }) => Option.some(providers),
    refreshing: ({ providers }) => Option.some(providers),
    degraded: ({ providers }) => Option.some(providers),
    unavailable: ({ providers }) => Option.some(providers),
  })))
  const catalogFailures = Option.getOrElse(Option.map(catalogState, (state) => ModelCatalogLifecycle.match(state, {
    loading: () => [] as readonly ProviderCatalogFailure[],
    ready: () => [] as readonly ProviderCatalogFailure[],
    refreshing: ({ failures }) => failures,
    degraded: ({ failures }) => failures,
    unavailable: ({ failures }) => failures,
  })), () => [] as readonly ProviderCatalogFailure[])
  const slotsSnapshot = modelConfig ? Result.value(modelConfig.slots) : Option.none()
  const slotConfig = Option.getOrNull(Option.flatMap(slotsSnapshot, ({ state }) => ModelSlotsLifecycle.match(state, {
    loading: () => Option.none(),
    ready: ({ config }) => Option.some(config.slots),
    refreshing: ({ config }) => Option.some(config.slots),
    degraded: ({ config }) => Option.some(config.slots),
    unavailable: ({ config }) => Option.some(config.slots),
  })))
  const catalogLoading = Option.match(catalogState, {
    onNone: () => modelConfig !== undefined && !Result.isFailure(modelConfig.catalog),
    onSome: (state) => ModelCatalogLifecycle.is(state, 'loading'),
  })
  const catalogRefreshing = modelConfig !== undefined && (
    Result.isWaiting(modelConfig.catalogRefresh)
    || Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'refreshing'))
  )

  const [updateHovered, setUpdateHovered] = useState(false)
  const [disconnectHovered, setDisconnectHovered] = useState(false)
  const [saveHovered, setSaveHovered] = useState(false)
  const [cancelHovered, setCancelHovered] = useState(false)
  const [confirmHovered, setConfirmHovered] = useState(false)
  const [refreshHovered, setRefreshHovered] = useState(false)
  const [localSetupHovered, setLocalSetupHovered] = useState(false)
  const [cloudSetupHovered, setCloudSetupHovered] = useState(false)

  // Dropdown state
  const [dropdownTarget, setDropdownTarget] = useState<DropdownTarget>(null)
  const [dropdownIndex, setDropdownIndex] = useState(0)

  // Per-slot hover state for model and thinking dropdown boxes
  const [modelHovered, setModelHovered] = useState<Record<string, boolean>>({})
  const [thinkingHovered, setThinkingHovered] = useState<Record<string, boolean>>({})
  const [resetHovered, setResetHovered] = useState(false)

  const beginEdit = useCallback(() => {
    setInputValue('')
    setError(null)
    setMode('edit')
  }, [])

  const beginDisconnect = useCallback(() => {
    setError(null)
    setMode('confirm-disconnect')
  }, [])

  const cancelInline = useCallback(() => {
    setInputValue('')
    setError(null)
    setMode('view')
  }, [])

  const handleSave = useCallback(() => {
    if (auth.saving) return
    const trimmed = inputValue.trim()
    if (!trimmed) { setError('API key is required'); return }
    setPendingAuthAction('save')
    auth.save(trimmed)
  }, [auth, inputValue])

  const handleConfirmDisconnect = useCallback(() => {
    if (auth.saving) return
    setPendingAuthAction('clear')
    auth.clear()
  }, [auth])

  const authCompletionAtom = useMemo(
    () => Atom.make(Effect.sync(() => {
      if (!pendingAuthAction || auth.saving) return
      if (auth.error) {
        setPendingAuthAction(null)
        return
      }
      const completed = pendingAuthAction === 'save'
        ? auth.source === 'config'
        : auth.source === 'none'
      if (!completed) return
      setPendingAuthAction(null)
      setInputValue('')
      setError(null)
      setMode('view')
    })),
    [auth.error, auth.saving, auth.source, pendingAuthAction],
  )
  useAtomMount(authCompletionAtom)

  const dropdownItems = useMemo(() => {
    if (!dropdownTarget) return []
    if (dropdownTarget.field === 'model') {
      return (models ?? []).map(m => ({
        id: m.providerModelId,
        providerId: m.providerId,
        label: `${m.displayName} · ${formatContextWindow(m.contextWindow)} ctx${m.pricing ? ` · ${formatPricing(m.pricing)}` : ''}${m.availability._tag === 'Disabled' ? ` · ${disabledReasonLabel(m.availability.reason)}` : ''}`,
        disabled: m.availability._tag === 'Disabled',
      }))
    }
    return REASONING_OPTIONS.map(o => ({ id: o.value, label: o.label }))
  }, [dropdownTarget, models])

  const openDropdown = useCallback((target: DropdownTarget) => {
    setDropdownTarget(target)
    setDropdownIndex(0)
  }, [])

  const closeDropdown = useCallback(() => {
    setDropdownTarget(null)
  }, [])

  const selectDropdownItem = useCallback((index: number) => {
    if (!dropdownTarget || !modelConfig) return
    if (dropdownTarget.field === 'model') {
      const model = models?.[index]
      if (!model || model.availability._tag === 'Disabled') return
      void modelConfig.updateSlotModel(dropdownTarget.slotId, model.providerId, model.providerModelId)
    } else {
      const option = REASONING_OPTIONS[index]
      if (!option) return
      void modelConfig.updateSlotReasoning(dropdownTarget.slotId, option.value)
    }
    closeDropdown()
  }, [dropdownTarget, modelConfig, models, closeDropdown])

  const unavailableProviders = providers?.filter((provider) =>
    provider.status && provider.status !== 'ok'
  ) ?? []

  useKeyboard(useCallback((key: KeyEvent) => {
    if (!isVisible) return

    if (dropdownTarget) {
      if (key.name === 'escape') { key.preventDefault(); closeDropdown(); return }
      if (key.name === 'up' || key.name === 'k') { key.preventDefault(); setDropdownIndex(i => Math.max(0, i - 1)); return }
      if (key.name === 'down' || key.name === 'j') { key.preventDefault(); setDropdownIndex(i => Math.min(dropdownItems.length - 1, i + 1)); return }
      if (key.name === 'return' || key.name === 'enter') { key.preventDefault(); selectDropdownItem(dropdownIndex); return }
      return
    }

    if (key.name === 'escape') {
      key.preventDefault()
      if (mode === 'edit' || mode === 'confirm-disconnect') { cancelInline(); return }
      onClose()
      return
    }
    const inferenceSourceAction = mode === 'view' ? getInferenceSourceAction(key.name) : null
    if (inferenceSourceAction) {
      key.preventDefault()
      if (inferenceSourceAction === 'local') onManageLocalModels()
      else onConfigureCloud()
      return
    }
    if (mode === 'edit' && (key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      handleSave()
    }
  }, [isVisible, dropdownTarget, dropdownItems, dropdownIndex, mode, onClose, onConfigureCloud, onManageLocalModels, cancelInline, handleSave, closeDropdown, selectDropdownItem]))

  if (!isVisible) return null

  return (
    <box style={{ position: 'relative', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Settings</span>
        </text>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>Esc to close</span>
        </text>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Magnitude section */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Magnitude</span>
        </text>
      </box>

      {/* Status / inline controls */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingBottom: 1, flexShrink: 0, flexDirection: 'column' }}>
        {mode === 'view' && (auth.source === 'env' || auth.source === 'env-local') && (
          <>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.success }}>{'● Connected '}</text>
              <text style={{ fg: theme.muted }}>{`via ${auth.envVarName} `}</text>
              {auth.key && (
                <text style={{ fg: theme.foreground }}>
                  <span attributes={TextAttributes.DIM}>{`(${maskApiKey(auth.key)})`}</span>
                </text>
              )}
            </box>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>To change this key, update the env var and relaunch.</span>
            </text>
          </>
        )}

        {mode === 'view' && auth.source === 'config' && (
          <box style={{ flexDirection: 'column' }}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.success }}>{'● Connected '}</text>
              {auth.maskedKey && (
                <text style={{ fg: theme.foreground }}>
                  <span attributes={TextAttributes.DIM}>{`(${auth.maskedKey})`}</span>
                </text>
              )}
            </box>
            <box style={{ flexDirection: 'row', paddingTop: 1 }}>
              <Button onClick={beginEdit} onMouseOver={() => setUpdateHovered(true)} onMouseOut={() => setUpdateHovered(false)}>
                <text style={{ fg: updateHovered ? theme.foreground : theme.muted }}>{'[Update key]'}</text>
              </Button>
              <text> </text>
              <Button onClick={beginDisconnect} onMouseOver={() => setDisconnectHovered(true)} onMouseOut={() => setDisconnectHovered(false)}>
                <text style={{ fg: disconnectHovered ? theme.foreground : theme.muted }}>{'[Disconnect]'}</text>
              </Button>
            </box>
          </box>
        )}

        {mode === 'view' && auth.source === 'none' && (
          <box style={{ flexDirection: 'row' }}>
            <text style={{ fg: theme.muted }}>{'○ Not connected '}</text>
            <text style={{ fg: theme.muted }}>{'· '}</text>
            <Button onClick={beginEdit} onMouseOver={() => setUpdateHovered(true)} onMouseOut={() => setUpdateHovered(false)}>
              <text style={{ fg: updateHovered ? theme.foreground : theme.muted }}>{'[Set API key]'}</text>
            </Button>
          </box>
        )}

        {mode === 'edit' && (
          <box style={{ flexDirection: 'column' }}>
            <box style={{ borderStyle: 'single', borderColor: displayedAuthError ? theme.error : theme.primary, paddingLeft: 1, paddingRight: 1, flexShrink: 0, width: 80 }}>
              <SingleLineInput value={inputValue} onChange={(v) => { setInputValue(v); setError(null) }} placeholder="Paste new API key" focused={true} />
            </box>
            {displayedAuthError && <box style={{ paddingTop: 1 }}><text style={{ fg: theme.error }}>{displayedAuthError}</text></box>}
            <box style={{ flexDirection: 'row', paddingTop: 1 }}>
              <Button onClick={handleSave} onMouseOver={() => setSaveHovered(true)} onMouseOut={() => setSaveHovered(false)}>
                <text style={{ fg: saveHovered ? theme.primary : theme.foreground }}>{auth.saving ? '[Saving...]' : '[Save]'}</text>
              </Button>
              <text> </text>
              <Button onClick={cancelInline} onMouseOver={() => setCancelHovered(true)} onMouseOut={() => setCancelHovered(false)}>
                <text style={{ fg: cancelHovered ? theme.foreground : theme.muted }}>{'[Cancel]'}</text>
              </Button>
            </box>
            <box style={{ paddingTop: 1 }}>
              <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>Enter to save, Esc to cancel</span></text>
            </box>
          </box>
        )}

        {mode === 'confirm-disconnect' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.foreground }}>Disconnect this key? You will need to set another to reconnect.</text>
            <box style={{ flexDirection: 'row', paddingTop: 1 }}>
              <Button onClick={handleConfirmDisconnect} onMouseOver={() => setConfirmHovered(true)} onMouseOut={() => setConfirmHovered(false)}>
                <text style={{ fg: confirmHovered ? theme.error : theme.foreground }}>{auth.saving ? '[Disconnecting...]' : '[Yes, disconnect]'}</text>
              </Button>
              <text> </text>
              <Button onClick={cancelInline} onMouseOver={() => setCancelHovered(true)} onMouseOut={() => setCancelHovered(false)}>
                <text style={{ fg: cancelHovered ? theme.foreground : theme.muted }}>{'[Cancel]'}</text>
              </Button>
            </box>
            {displayedAuthError && <box style={{ paddingTop: 1 }}><text style={{ fg: theme.error }}>{displayedAuthError}</text></box>}
          </box>
        )}
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Inference source setup */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column', flexShrink: 0 }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Inference sources</span>
        </text>
        <text style={{ fg: theme.muted }}>Download or attach local models, or configure Magnitude Cloud fallback.</text>
        <box style={{ flexDirection: 'row', paddingTop: 1 }}>
          <Button
            onClick={onManageLocalModels}
            onMouseOver={() => setLocalSetupHovered(true)}
            onMouseOut={() => setLocalSetupHovered(false)}
          >
            <text style={{ fg: localSetupHovered ? theme.primary : theme.muted }}>{`[${INFERENCE_SOURCE_ACTIONS.local.label} · ${INFERENCE_SOURCE_ACTIONS.local.key.toUpperCase()}]`}</text>
          </Button>
          <text> </text>
          <Button
            onClick={onConfigureCloud}
            onMouseOver={() => setCloudSetupHovered(true)}
            onMouseOut={() => setCloudSetupHovered(false)}
          >
            <text style={{ fg: cloudSetupHovered ? theme.primary : theme.muted }}>{`[${INFERENCE_SOURCE_ACTIONS.cloud.label} · ${INFERENCE_SOURCE_ACTIONS.cloud.key.toUpperCase()}]`}</text>
          </Button>
        </box>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Model Selection */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0, flexDirection: 'row' }}>
        <text style={{ fg: theme.foreground, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Model Selection</span>
        </text>
        {modelConfig && (
          <Button onClick={() => { void modelConfig.refreshModels() }} onMouseOver={() => setRefreshHovered(true)} onMouseOut={() => setRefreshHovered(false)}>
            <text style={{ fg: refreshHovered ? theme.primary : theme.muted }}>{catalogRefreshing ? '[Refreshing...]' : '[Refresh models]'}</text>
          </Button>
        )}
      </box>

      {/* Slot cards with inline dropdowns */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingBottom: 1, flexDirection: 'column', flexShrink: 0 }}>
        {unavailableProviders.map((provider) => (
          <box key={provider.id} style={{ paddingBottom: 1 }}>
            <text style={{ fg: provider.status === 'error' ? theme.error : theme.warning }}>
              {provider.status === 'not_found'
                ? `⚠ ${provider.displayName} not detected.${provider.hint ? ` ${provider.hint}` : ''}`
                : provider.status === 'loading'
                  ? `◐ ${provider.message ?? `${provider.displayName} is loading models...`}`
                  : `✗ ${provider.displayName}: ${provider.message ?? 'Unknown provider error'}`}
            </text>
          </box>
        ))}
        {modelConfig && Result.isFailure(modelConfig.catalog) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>
              {Option.isSome(catalogSnapshot)
                ? '⚠ Lost contact with the model catalog; showing the last received state.'
                : '✗ Unable to read the model catalog from the daemon.'}
            </text>
          </box>
        )}
        {catalogFailures.length > 0 && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'unavailable')) ? theme.error : theme.warning }}>
              {Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'unavailable'))
                ? `✗ Model catalog unavailable. ${formatCatalogFailures(catalogFailures)}`
                : `⚠ Showing available or last-known models. ${formatCatalogFailures(catalogFailures)}`}
            </text>
          </box>
        )}
        {modelConfig && Result.isFailure(modelConfig.catalogRefresh) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>✗ Failed to request a model catalog refresh.</text>
          </box>
        )}
        {modelConfig && Result.isFailure(modelConfig.slotUpdate) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>✗ Failed to update model configuration.</text>
          </box>
        )}
        {SLOT_IDS.map((slotId) => {
          const label = SLOT_DISPLAY_NAMES[slotId]
          const description = SLOT_DESCRIPTIONS[slotId]
          const currentOverride = slotConfig?.[slotId] ?? null
          const defaultEffort = DEFAULT_REASONING_EFFORT[slotId]
          const currentEffort = currentOverride?.reasoningEffort ?? defaultEffort
          const availableModels = models?.filter(m => m.availability._tag === 'Available') ?? null
          const defaultModel = availableModels?.find(m => m.slots?.includes(slotId)) ?? availableModels?.[0] ?? null
          const effectiveModel = currentOverride?.providerId && currentOverride.providerModelId
            ? models?.find(m =>
                m.providerId === currentOverride.providerId
                && m.providerModelId === currentOverride.providerModelId
              ) ?? null
            : currentOverride?.providerModelId
              ? models?.find(m => m.providerModelId === currentOverride.providerModelId) ?? null
              : defaultModel
          const modelLabel = effectiveModel
            ? effectiveModel.displayName
            : '—'
          const thinkingLabel = REASONING_OPTIONS.find(o => o.value === currentEffort)?.label ?? '—'
          const loading = catalogLoading
          const isThisDropdownOpen = dropdownTarget?.slotId === slotId

          return (
            <box key={slotId} style={{ flexDirection: 'column', paddingBottom: 1, position: 'relative', ...(isThisDropdownOpen ? { zIndex: 200 } : {}) }}>
              <text style={{ fg: theme.primary }}>
                <span attributes={TextAttributes.BOLD}>{label}</span>
              </text>

              {/* Dropdown boxes side by side */}
              <box style={{ flexDirection: 'row', paddingTop: 0, ...(isThisDropdownOpen ? { zIndex: 200 } : {}) }}>
                {/* Model dropdown — relative wrapper, dropdown floats below with absolute */}
                {(() => {
                  const w = 36; const pad = 2; const border = 2; const arrow = '▾'
                  const maxLen = w - pad - border - arrow.length - 1
                  const trunc = loading ? 'Loading...' : modelLabel.length > maxLen ? modelLabel.slice(0, maxLen - 1) + '…' : modelLabel
                  const padded = trunc + ' '.repeat(Math.max(0, maxLen - trunc.length))
                  const isOpen = isThisDropdownOpen && dropdownTarget?.field === 'model'
                  return (
                    <box style={{ position: 'relative', flexDirection: 'column', width: w, zIndex: 200 }}>
                      <Button
                        onClick={() => isOpen ? closeDropdown() : openDropdown({ slotId, field: 'model' })}
                        onMouseOver={() => setModelHovered(prev => ({ ...prev, [slotId]: true }))}
                        onMouseOut={() => setModelHovered(prev => ({ ...prev, [slotId]: false }))}
                        style={{
                          borderStyle: 'rounded',
                          borderColor: isOpen || modelHovered[slotId] ? theme.primary : theme.border,
                          paddingLeft: 1, paddingRight: 1, width: w, flexDirection: 'row',
                        }}
                      >
                        <text style={{ fg: isOpen || modelHovered[slotId] ? theme.primary : theme.foreground, flexGrow: 1 }}>{padded}</text>
                        <text style={{ fg: isOpen || modelHovered[slotId] ? theme.primary : theme.muted }}>{arrow}</text>
                      </Button>
                      {isOpen && (
                        <box style={{
                          position: 'absolute',
                          top: 3,
                          left: 0,
                          zIndex: 200,
                          flexDirection: 'column',
                          borderStyle: 'rounded',
                          borderColor: theme.primary,
                          width: w,
                          backgroundColor: theme.terminalDetectedBg,
                        }}>
                          {dropdownItems.length === 0 ? (
                            <text style={{ fg: Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'unavailable')) ? theme.error : theme.muted }}>
                              <span attributes={TextAttributes.DIM}>{Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'unavailable')) ? 'Catalog unavailable' : 'No models'}</span>
                            </text>
                          ) : dropdownItems.map((item, index) => {
                            const sel = index === dropdownIndex
                            const itemDisabled = 'disabled' in item && item.disabled
                            return (
                              <Button key={`${'providerId' in item ? item.providerId : 'model'}:${item.id}`} onClick={() => { if (!itemDisabled) selectDropdownItem(index) }} onMouseOver={() => setDropdownIndex(index)}
                                style={{ flexDirection: 'row', width: w - 2, backgroundColor: theme.terminalDetectedBg }}>
                                <text style={{ fg: itemDisabled ? theme.muted : sel ? theme.primary : theme.foreground, overflow: 'hidden' }}>
                                  {sel ? '▸ ' : '  '}{item.label.length > maxLen - 2 ? item.label.slice(0, maxLen - 3) + '…' : item.label}
                                </text>
                              </Button>
                            )
                          })}
                        </box>
                      )}
                    </box>
                  )
                })()}

                <text> </text>

                {/* Thinking dropdown — relative wrapper, dropdown floats below with absolute */}
                {(() => {
                  const w = 18; const pad = 2; const border = 2; const arrow = '▾'
                  const fullLabel = thinkingLabel + (currentOverride?.reasoningEffort === undefined ? ' (def)' : '')
                  const maxLen = w - pad - border - arrow.length - 1
                  const trunc = fullLabel.length > maxLen ? fullLabel.slice(0, maxLen - 1) + '…' : fullLabel
                  const padded = trunc + ' '.repeat(Math.max(0, maxLen - trunc.length))
                  const isOpen = isThisDropdownOpen && dropdownTarget?.field === 'thinking'
                  return (
                    <box style={{ position: 'relative', flexDirection: 'column', width: w, zIndex: 200 }}>
                      <Button
                        onClick={() => isOpen ? closeDropdown() : openDropdown({ slotId, field: 'thinking' })}
                        onMouseOver={() => setThinkingHovered(prev => ({ ...prev, [slotId]: true }))}
                        onMouseOut={() => setThinkingHovered(prev => ({ ...prev, [slotId]: false }))}
                        style={{
                          borderStyle: 'rounded',
                          borderColor: isOpen || thinkingHovered[slotId] ? theme.primary : theme.border,
                          paddingLeft: 1, paddingRight: 1, width: w, flexDirection: 'row',
                        }}
                      >
                        <text style={{ fg: isOpen || thinkingHovered[slotId] ? theme.primary : theme.foreground, flexGrow: 1 }}>{padded}</text>
                        <text style={{ fg: isOpen || thinkingHovered[slotId] ? theme.primary : theme.muted }}>{arrow}</text>
                      </Button>
                      {isOpen && (
                        <box style={{
                          position: 'absolute',
                          top: 3,
                          left: 0,
                          zIndex: 200,
                          flexDirection: 'column',
                          borderStyle: 'rounded',
                          borderColor: theme.primary,
                          width: w,
                          backgroundColor: theme.terminalDetectedBg,
                        }}>
                          {dropdownItems.map((item, index) => {
                            const sel = index === dropdownIndex
                            return (
                              <Button key={item.id} onClick={() => selectDropdownItem(index)} onMouseOver={() => setDropdownIndex(index)}
                                style={{ flexDirection: 'row', width: w - 2, backgroundColor: theme.terminalDetectedBg }}>
                                <text style={{ fg: sel ? theme.primary : theme.foreground }}>
                                  {sel ? '▸ ' : '  '}{item.label}
                                </text>
                              </Button>
                            )
                          })}
                        </box>
                      )}
                    </box>
                  )
                })()}
              </box>

              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>{description}</span>
              </text>
            </box>
          )
        })}

        {modelConfig && (
          <box style={{ paddingTop: 1 }}>
            <Button
              onClick={() => { void modelConfig.resetToDefaults() }}
              onMouseOver={() => setResetHovered(true)}
              onMouseOut={() => setResetHovered(false)}
            >
              <text style={{ fg: resetHovered ? theme.foreground : theme.muted }}>
                {'[Reset to defaults]'}
              </text>
            </Button>
          </box>
        )}
      </box>

    </box>
  )
})

export type { SettingsOverlayProps }
