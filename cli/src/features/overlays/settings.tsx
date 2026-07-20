import { memo, useCallback, useState, useMemo } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { Atom, Result, useAtomMount } from '@effect-atom/atom-react'
import { Effect, Option } from 'effect'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { SingleLineInput } from '../composer/single-line-input'
import type { AuthInfo } from './auth-display'
import { reasoningEffortControl, reasoningPropertyLabel, selectedSlotModel, useLocalInferenceQuery, visionPropertyLabel, type UseModelConfigResult } from '@magnitudedev/client-common'
import { ModelCatalogLifecycle, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS, type ProviderCatalogFailure, type SlotId } from '@magnitudedev/sdk'
import { getInferenceSourceAction, INFERENCE_SOURCE_ACTIONS } from './inference-source-actions'
import { getCatalogFailureNotice } from './catalog-failure-notice'
import { describeLocalHardware } from '../local-inference/view-model'
import { writeTextToClipboard } from '../../utils/clipboard'
import { BOX_CHARS } from '../../utils/ui-constants'

const MAGNITUDE_CLOUD_URL = 'https://app.magnitude.dev'
const SETTINGS_SECTION_WIDTH = 72
const SETTINGS_SECTION_LABEL_GAP = 2
const settingsSectionRule = (label: string): string =>
  '─'.repeat(Math.max(0, SETTINGS_SECTION_WIDTH - label.length - SETTINGS_SECTION_LABEL_GAP))

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
}

type Mode = 'view' | 'edit' | 'confirm-disconnect'
type PendingAuthAction = 'save' | 'clear' | null

type DropdownTarget =
  | { slotId: SlotId; field: 'model' }
  | { slotId: SlotId; field: 'thinking' }
  | null

function formatContextWindow(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

function formatPricing(pricing: { input: number; output: number; cachedInput?: number }): string {
  return `$${pricing.input.toFixed(2)}/$${pricing.output.toFixed(2)}`
}

const disabledReasonLabel = (reason: "insufficient_resources" | "provider_unavailable" | "model_unavailable" | "installation_unavailable" | "incompatible_runtime" | "invalid_configuration"): string => ({
  insufficient_resources: 'not enough free memory',
  provider_unavailable: 'server unavailable',
  model_unavailable: 'model unavailable',
  installation_unavailable: 'local inference unavailable',
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
}: SettingsOverlayProps) {
  const theme = useTheme()
  const localInferenceState = useLocalInferenceQuery()
  const localInferenceSnapshot = Result.value(localInferenceState)
  const host = Option.flatMap(localInferenceSnapshot, (state) =>
    state.host._tag === 'Available' ? Option.some(state.host.profile) : Option.none()
  )
  const hardware = Option.map(host, describeLocalHardware)
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
  const slotsState = Option.map(slotsSnapshot, ({ state }) => state)
  const catalogLoading = Option.match(catalogState, {
    onNone: () => modelConfig !== undefined && !Result.isFailure(modelConfig.catalog),
    onSome: (state) => ModelCatalogLifecycle.is(state, 'loading'),
  })
  const catalogRefreshing = modelConfig !== undefined && (
    Result.isWaiting(modelConfig.catalogRefresh)
    || Option.exists(catalogState, (state) => ModelCatalogLifecycle.is(state, 'refreshing'))
  )
  const catalogUnavailable = Option.exists(
    catalogState,
    (state) => ModelCatalogLifecycle.is(state, 'unavailable'),
  )
  const catalogFailureNotice = getCatalogFailureNotice(catalogFailures, catalogUnavailable)
  const noModelsConfigured = slots.every((slot) => slot.modelDisplayName === null)

  const [updateHovered, setUpdateHovered] = useState(false)
  const [disconnectHovered, setDisconnectHovered] = useState(false)
  const [saveHovered, setSaveHovered] = useState(false)
  const [cancelHovered, setCancelHovered] = useState(false)
  const [confirmHovered, setConfirmHovered] = useState(false)
  const [refreshHovered, setRefreshHovered] = useState(false)
  const [localSetupHovered, setLocalSetupHovered] = useState(false)
  const [copyCloudLinkHovered, setCopyCloudLinkHovered] = useState(false)
  const [cloudLinkCopied, setCloudLinkCopied] = useState(false)

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

  const copyCloudLink = useCallback(() => {
    void writeTextToClipboard(MAGNITUDE_CLOUD_URL).then((copied) => {
      if (!copied) return
      setCloudLinkCopied(true)
      setTimeout(() => setCloudLinkCopied(false), 2_000)
    })
  }, [])

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

  const selectedForSlot = useCallback((slotId: SlotId) => Option.flatMap(
    Option.all({ catalog: catalogState, slots: slotsState }),
    ({ catalog, slots }) => selectedSlotModel(catalog, slots, slotId),
  ), [catalogState, slotsState])

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
    return Option.match(selectedForSlot(dropdownTarget.slotId), {
      onNone: () => [],
      onSome: ({ model }) => {
        const control = reasoningEffortControl(model)
        return control._tag === 'Available'
          ? control.options.map((option) => ({ id: option.value, label: option.label }))
          : []
      },
    })
  }, [dropdownTarget, models, selectedForSlot])

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
      const option = dropdownItems[index]
      if (!option) return
      void modelConfig.updateSlotReasoning(dropdownTarget.slotId, option.id)
    }
    closeDropdown()
  }, [dropdownItems, dropdownTarget, modelConfig, models, closeDropdown])

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
      onManageLocalModels()
      return
    }
    if (mode === 'edit' && (key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      handleSave()
    }
  }, [isVisible, dropdownTarget, dropdownItems, dropdownIndex, mode, onClose, onManageLocalModels, cancelInline, handleSave, closeDropdown, selectDropdownItem]))

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

      {/* Detected hardware */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column', flexShrink: 0 }}>
        <box style={{ flexDirection: 'row', paddingBottom: 1, width: '100%', maxWidth: SETTINGS_SECTION_WIDTH }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>DETECTED HARDWARE</text>
          <text style={{ fg: theme.border }}>  {settingsSectionRule('DETECTED HARDWARE')}</text>
        </box>
        {Option.match(hardware, {
          onNone: () => (
            <text style={{ fg: Result.isFailure(localInferenceState) ? theme.warning : theme.muted }}>
              {Result.isFailure(localInferenceState) ? 'Hardware detection unavailable' : 'Detecting hardware…'}
            </text>
          ),
          onSome: (detected) => (
            <box style={{
              paddingLeft: 1,
              paddingRight: 1,
              flexDirection: 'column',
              width: '100%',
              maxWidth: SETTINGS_SECTION_WIDTH,
            }}>
              <text style={{ fg: theme.foreground }}><span attributes={TextAttributes.BOLD}>{detected.system.name}</span></text>
              {detected.system.details.map((detail) => <text key={detail} style={{ fg: theme.muted }}>{detail}</text>)}
              {detected.accelerators.map((accelerator) => (
                <box key={`${accelerator.name}:${accelerator.details}`} style={{ flexDirection: 'column', paddingTop: 1 }}>
                  <text style={{ fg: theme.foreground }}>{accelerator.name}</text>
                  <text style={{ fg: theme.muted }}>{accelerator.details}</text>
                </box>
              ))}
              {detected.accelerators.length === 0 && Option.exists(host, (profile) => !profile.memoryDomains.some((domain) => domain.kind === 'unified_memory')) && (
                <text style={{ fg: theme.muted }}>CPU inference · No GPU detected</text>
              )}
            </box>
          ),
        })}
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>{'─'.repeat(60)}</text>
      </box>

      {/* Magnitude Cloud section */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.foreground }}>
          <span attributes={TextAttributes.BOLD}>Magnitude Cloud</span>
        </text>
      </box>

      {/* Status / inline controls */}
      <box style={{ paddingLeft: 2, paddingRight: 2, paddingBottom: 1, flexShrink: 0, flexDirection: 'column' }}>
        {auth.source === 'none' && (
          <text style={{ fg: theme.muted }}>○ Not connected</text>
        )}

        {mode === 'view' && auth.source === 'env' && (
          <>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.success }}>{`● Connected via ${auth.envVarName} `}</text>
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
              <text style={{ fg: theme.success }}>{'● Connected via API key '}</text>
              {auth.maskedKey && (
                <text style={{ fg: theme.foreground }}>
                  <span attributes={TextAttributes.DIM}>{`(${auth.maskedKey})`}</span>
                </text>
              )}
            </box>
            <box style={{ flexDirection: 'row', paddingTop: 1 }}>
              <Button onClick={beginEdit} onMouseOver={() => setUpdateHovered(true)} onMouseOut={() => setUpdateHovered(false)}>
                <text style={{ fg: updateHovered ? theme.foreground : theme.muted }}>{'[Update API key]'}</text>
              </Button>
              <text> </text>
              <Button onClick={beginDisconnect} onMouseOver={() => setDisconnectHovered(true)} onMouseOut={() => setDisconnectHovered(false)}>
                <text style={{ fg: disconnectHovered ? theme.foreground : theme.muted }}>{'[Disconnect]'}</text>
              </Button>
            </box>
          </box>
        )}

        {mode === 'view' && auth.source === 'none' && (
          <box style={{ flexDirection: 'column' }}>
            <box style={{ paddingTop: 1 }}>
              <Button
                onClick={beginEdit}
                onMouseOver={() => setUpdateHovered(true)}
                onMouseOut={() => setUpdateHovered(false)}
                style={{
                  borderStyle: 'single',
                  customBorderChars: BOX_CHARS,
                  borderColor: updateHovered ? theme.primary : theme.border,
                  paddingLeft: 1,
                  paddingRight: 1,
                  width: 15,
                }}
              >
                <text style={{ fg: updateHovered ? theme.primary : theme.foreground }}>Add API Key</text>
              </Button>
            </box>
            <box style={{ flexDirection: 'row', paddingTop: 1 }}>
              <text style={{ fg: theme.muted }}>Get an API key → </text>
              <text style={{ fg: theme.primary }}>{MAGNITUDE_CLOUD_URL}</text>
              <text> </text>
              <Button
                onClick={copyCloudLink}
                onMouseOver={() => setCopyCloudLinkHovered(true)}
                onMouseOut={() => setCopyCloudLinkHovered(false)}
              >
                <text style={{ fg: cloudLinkCopied ? theme.success : copyCloudLinkHovered ? theme.foreground : theme.muted }}>
                  {cloudLinkCopied ? '[Copied ✓]' : '[Copy link]'}
                </text>
              </Button>
            </box>
          </box>
        )}

        {mode === 'edit' && (
          <box style={{ flexDirection: 'column' }}>
            <box style={{ borderStyle: 'single', borderColor: displayedAuthError ? theme.error : theme.primary, paddingLeft: 1, paddingRight: 1, flexShrink: 0, width: 80 }}>
              <SingleLineInput value={inputValue} onChange={(v) => { setInputValue(v); setError(null) }} placeholder="Paste Magnitude Cloud API key" focused={true} />
            </box>
            <box style={{ flexDirection: 'row' }}>
              <Button onClick={handleSave} onMouseOver={() => setSaveHovered(true)} onMouseOut={() => setSaveHovered(false)}>
                <text style={{ fg: saveHovered ? theme.primary : theme.foreground }}>{auth.saving ? '[Saving...]' : '[Save (Enter)]'}</text>
              </Button>
              <text> </text>
              <Button onClick={cancelInline} onMouseOver={() => setCancelHovered(true)} onMouseOut={() => setCancelHovered(false)}>
                <text style={{ fg: cancelHovered ? theme.foreground : theme.muted }}>{'[Cancel (Esc)]'}</text>
              </Button>
            </box>
            {displayedAuthError && <box style={{ paddingTop: 1 }}><text style={{ fg: theme.error }}>{displayedAuthError}</text></box>}
            {auth.source === 'none' && (
              <box style={{ flexDirection: 'row', paddingTop: 1 }}>
                <text style={{ fg: theme.muted }}>Get an API key → </text>
                <text style={{ fg: theme.primary }}>{MAGNITUDE_CLOUD_URL}</text>
                <text> </text>
                <Button
                  onClick={copyCloudLink}
                  onMouseOver={() => setCopyCloudLinkHovered(true)}
                  onMouseOut={() => setCopyCloudLinkHovered(false)}
                >
                  <text style={{ fg: cloudLinkCopied ? theme.success : copyCloudLinkHovered ? theme.foreground : theme.muted }}>
                    {cloudLinkCopied ? '[Copied ✓]' : '[Copy link]'}
                  </text>
                </Button>
              </box>
            )}
          </box>
        )}

        {mode === 'confirm-disconnect' && (
          <box style={{ flexDirection: 'column' }}>
            <text style={{ fg: theme.foreground }}>Disconnect Magnitude Cloud? Cloud models will no longer be available.</text>
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
        <text style={{ fg: theme.muted }}>Download or manage local models.</text>
        <box style={{ flexDirection: 'row', paddingTop: 1 }}>
          <Button
            onClick={onManageLocalModels}
            onMouseOver={() => setLocalSetupHovered(true)}
            onMouseOut={() => setLocalSetupHovered(false)}
          >
            <text style={{ fg: localSetupHovered ? theme.primary : theme.muted }}>{`[${INFERENCE_SOURCE_ACTIONS.local.label} · ${INFERENCE_SOURCE_ACTIONS.local.key.toUpperCase()}]`}</text>
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
                ? `${provider.displayName} not detected.${provider.hint ? ` ${provider.hint}` : ''}`
                : provider.status === 'loading'
                  ? provider.message ?? `${provider.displayName} is loading models...`
                  : `${provider.displayName}: ${provider.message ?? 'Unknown provider error'}`}
            </text>
          </box>
        ))}
        {modelConfig && Result.isFailure(modelConfig.catalog) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>
              {Option.isSome(catalogSnapshot)
                ? 'Lost contact with the model catalog; showing the last received state.'
                : 'Unable to read the model catalog from the daemon.'}
            </text>
          </box>
        )}
        {noModelsConfigured && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground }}>Warning: No providers connected (local or cloud)</text>
          </box>
        )}
        {catalogFailureNotice && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{
              fg: catalogFailureNotice.tone === 'error' ? theme.error : theme.warning,
            }}>
              {catalogFailureNotice.message}
            </text>
          </box>
        )}
        {modelConfig && Result.isFailure(modelConfig.catalogRefresh) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>Failed to request a model catalog refresh.</text>
          </box>
        )}
        {modelConfig && Result.isFailure(modelConfig.slotUpdate) && (
          <box style={{ paddingBottom: 1 }}>
            <text style={{ fg: theme.error }}>Failed to update model configuration.</text>
          </box>
        )}
        {SLOT_IDS.map((slotId) => {
          const label = SLOT_DISPLAY_NAMES[slotId]
          const description = SLOT_DESCRIPTIONS[slotId]
          const selected = selectedForSlot(slotId)
          const modelLabel = Option.match(selected, { onNone: () => '—', onSome: ({ model }) => model.displayName })
          const thinkingLabel = Option.match(selected, {
            onNone: () => '—',
            onSome: ({ model, slot }) => {
              const control = reasoningEffortControl(model)
              return control._tag === 'Available'
                ? control.options.find((option) => option.value === slot.selection.reasoningEffort)?.label ?? '—'
                : control.label
            },
          })
          const thinkingAvailable = Option.exists(selected, ({ model }) => reasoningEffortControl(model)._tag === 'Available')
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
                            <text style={{ fg: catalogFailureNotice?.tone === 'error' ? theme.error : theme.muted }}>
                              <span attributes={TextAttributes.DIM}>{catalogFailureNotice?.tone === 'error' ? 'Catalog unavailable' : 'No models configured'}</span>
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
                  const fullLabel = thinkingLabel
                  const maxLen = w - pad - border - arrow.length - 1
                  const trunc = fullLabel.length > maxLen ? fullLabel.slice(0, maxLen - 1) + '…' : fullLabel
                  const padded = trunc + ' '.repeat(Math.max(0, maxLen - trunc.length))
                  const isOpen = isThisDropdownOpen && dropdownTarget?.field === 'thinking'
                  return (
                    <box style={{ position: 'relative', flexDirection: 'column', width: w, zIndex: 200 }}>
                      <Button
                        onClick={() => {
                          if (thinkingAvailable) isOpen ? closeDropdown() : openDropdown({ slotId, field: 'thinking' })
                        }}
                        onMouseOver={() => setThinkingHovered(prev => ({ ...prev, [slotId]: true }))}
                        onMouseOut={() => setThinkingHovered(prev => ({ ...prev, [slotId]: false }))}
                        style={{
                          borderStyle: 'rounded',
                          borderColor: thinkingAvailable && (isOpen || thinkingHovered[slotId]) ? theme.primary : theme.border,
                          paddingLeft: 1, paddingRight: 1, width: w, flexDirection: 'row',
                        }}
                      >
                        <text style={{ fg: thinkingAvailable && (isOpen || thinkingHovered[slotId]) ? theme.primary : thinkingAvailable ? theme.foreground : theme.muted, flexGrow: 1 }}>{padded}</text>
                        <text style={{ fg: thinkingAvailable && (isOpen || thinkingHovered[slotId]) ? theme.primary : theme.muted }}>{arrow}</text>
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

              {Option.isSome(selected) && (
                <text style={{ fg: theme.muted }}>
                  <span attributes={TextAttributes.DIM}>{visionPropertyLabel(selected.value.model)} · {reasoningPropertyLabel(selected.value.model)}</span>
                </text>
              )}

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
