import { memo, useCallback, useMemo, useState } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../../hooks/use-theme'
import { Button } from '../../components/button'
import { SingleLineInput } from '../composer/single-line-input'
import type { ModelOption, UseModelConfigResult } from '@magnitudedev/client-common'
import {
  DEFAULT_REASONING_EFFORT,
  resolveReasoningEffort,
  SLOT_DESCRIPTIONS,
  SLOT_DISPLAY_NAMES,
  SLOT_IDS,
  type ProviderAuthSummary,
  type ProviderInfo,
  type SlotId,
} from '@magnitudedev/sdk'

interface SettingsOverlayProps {
  isVisible: boolean
  onClose: () => void
  providerAuths: readonly ProviderAuthSummary[] | null
  onSaveProviderApiKey: (providerId: string, key: string) => Promise<void>
  onDisconnectProvider: (providerId: string) => Promise<void>
  modelConfig?: UseModelConfigResult
}

type DropdownTarget =
  | { slotId: SlotId; field: 'model' }
  | { slotId: SlotId; field: 'thinking' }
  | null

interface ModelPickerItem {
  id: string
  providerId: string
  providerName: string
  label: string
  rawId: string
}

interface ThinkingPickerItem {
  id: string
  label: string
  value: string | null
}

const PICKER_WINDOW_SIZE = 6

function formatContextWindow(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

function formatLabel(value: string): string {
  if (value === 'default') return 'Provider default'
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function effectiveModelForSlot(
  slotId: SlotId,
  modelConfig: UseModelConfigResult | undefined,
): ModelOption | null {
  const models = selectableModels(modelConfig)
  const override = modelConfig?.slotConfig?.[slotId]
  if (override?.providerId && override.providerModelId) {
    return models.find((model) =>
      model.providerId === override.providerId
      && model.providerModelId === override.providerModelId
    ) ?? null
  }
  return models.find((model) => model.slots?.includes(slotId)) ?? models[0] ?? null
}

function selectableModels(modelConfig: UseModelConfigResult | undefined): readonly ModelOption[] {
  const healthyProviderIds = new Set((modelConfig?.providers ?? [])
    .filter((provider) =>
      provider.authStatus !== 'not_configured'
      && (provider.status === undefined || provider.status === 'ok')
    )
    .map((provider) => provider.id))
  return (modelConfig?.models ?? []).filter((model) => healthyProviderIds.has(model.providerId))
}

function ProviderRows({
  providers,
  auths,
  editingProviderId,
  disconnectProviderId,
  inputValue,
  submitting,
  error,
  onBeginEdit,
  onBeginDisconnect,
  onInputChange,
  onSave,
  onDisconnect,
  onRefresh,
  refreshing,
  onCancel,
}: {
  providers: readonly ProviderInfo[] | null
  auths: readonly ProviderAuthSummary[] | null
  editingProviderId: string | null
  disconnectProviderId: string | null
  inputValue: string
  submitting: boolean
  error: string | null
  onBeginEdit: (providerId: string) => void
  onBeginDisconnect: (providerId: string) => void
  onInputChange: (value: string) => void
  onSave: () => void
  onDisconnect: () => void
  onRefresh: (providerId: string) => void
  refreshing: boolean
  onCancel: () => void
}) {
  const theme = useTheme()
  if (providers === null) return <text style={{ fg: theme.muted }}>Loading providers...</text>

  return (
    <>
      {providers.map((provider) => {
        const auth = auths?.find((candidate) => candidate.providerId === provider.id)
        const endpointProvider = provider.authKind === 'endpoint'
        const configured = auth?.configured === true || provider.authStatus === 'authenticated'
        const envManaged = auth?.source === 'env'
        const editing = editingProviderId === provider.id
        const confirming = disconnectProviderId === provider.id
        const connectionFailed = provider.status === 'error' || provider.status === 'not_found'
        const status = endpointProvider
          ? provider.status === 'ok' ? 'Running' : provider.status === 'loading' ? 'Loading' : 'Unavailable'
          : connectionFailed
            ? 'Connection error'
            : configured
              ? `${provider.status === 'ok' ? 'Connected' : 'Configured'}${auth?.source && auth.source !== 'none' ? ` via ${auth.source}` : ''}`
              : 'Not connected'
        const healthy = configured && !connectionFailed && (!endpointProvider || provider.status === 'ok')
        const statusColor = connectionFailed ? theme.error : healthy ? theme.success : theme.muted

        return (
          <box key={provider.id} style={{ flexDirection: 'column', paddingBottom: 1 }}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.foreground, width: 22 }}><span attributes={TextAttributes.BOLD}>{provider.displayName}</span></text>
              <text style={{ fg: statusColor, flexGrow: 1 }}>{configured ? '● ' : '○ '}{status}</text>
              {!endpointProvider && !envManaged && !editing && !confirming && (
                <box style={{ flexDirection: 'row' }}>
                  <Button onClick={() => onBeginEdit(provider.id)}><text style={{ fg: theme.primary }}>{configured ? '[Update]' : '[Connect]'}</text></Button>
                  {configured && <><text> </text><Button onClick={() => onBeginDisconnect(provider.id)}><text style={{ fg: theme.muted }}>[Disconnect]</text></Button></>}
                </box>
              )}
              {configured && !editing && !confirming && <><text> </text><Button onClick={() => onRefresh(provider.id)}><text style={{ fg: theme.muted }}>{refreshing ? '[...]' : '[Refresh]'}</text></Button></>}
            </box>
            {(auth?.maskedKey || auth?.endpoint) && (
              <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>{auth.maskedKey ?? auth.endpoint}</span></text>
            )}
            {provider.message && <text style={{ fg: theme.error }}>{provider.message}</text>}
            {editing && (
              <box style={{ flexDirection: 'column', paddingTop: 1 }}>
                <box style={{ borderStyle: 'single', borderColor: error ? theme.error : theme.primary, paddingLeft: 1, paddingRight: 1, width: 72 }}>
                  <SingleLineInput value={inputValue} onChange={onInputChange} placeholder="API key" focused />
                </box>
                <box style={{ flexDirection: 'row', paddingTop: 1 }}>
                  <Button onClick={onSave}><text style={{ fg: theme.primary }}>{submitting ? '[Saving...]' : '[Save]'}</text></Button>
                  <text> </text>
                  <Button onClick={onCancel}><text style={{ fg: theme.muted }}>[Cancel]</text></Button>
                </box>
              </box>
            )}
            {confirming && (
              <box style={{ flexDirection: 'row', paddingTop: 1 }}>
                <Button onClick={onDisconnect}><text style={{ fg: theme.error }}>{submitting ? '[Disconnecting...]' : '[Confirm disconnect]'}</text></Button>
                <text> </text>
                <Button onClick={onCancel}><text style={{ fg: theme.muted }}>[Cancel]</text></Button>
              </box>
            )}
            {(editing || confirming) && error && <text style={{ fg: theme.error }}>{error}</text>}
          </box>
        )
      })}
    </>
  )
}

export const SettingsOverlay = memo(function SettingsOverlay({
  isVisible,
  onClose,
  providerAuths,
  onSaveProviderApiKey,
  onDisconnectProvider,
  modelConfig,
}: SettingsOverlayProps) {
  const theme = useTheme()
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null)
  const [disconnectProviderId, setDisconnectProviderId] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dropdownTarget, setDropdownTarget] = useState<DropdownTarget>(null)
  const [dropdownIndex, setDropdownIndex] = useState(0)
  const [dropdownQuery, setDropdownQuery] = useState('')
  const [refreshHovered, setRefreshHovered] = useState(false)
  const [resetHovered, setResetHovered] = useState(false)

  const providers = modelConfig?.providers ?? null

  const modelItems = useMemo((): readonly ModelPickerItem[] => {
    if (dropdownTarget?.field !== 'model') return []
    const query = dropdownQuery.trim().toLowerCase()
    const providerOrder = new Map((providers ?? []).map((provider, index) => [provider.id, index]))
    return selectableModels(modelConfig)
      .filter((model) => !query || [model.displayName, model.providerModelId, model.providerId, model.modelFamilyId].some((value) => value.toLowerCase().includes(query)))
      .map((model) => ({
        id: model.providerModelId,
        providerId: model.providerId,
        providerName: providers?.find((provider) => provider.id === model.providerId)?.displayName ?? model.providerId,
        label: `${model.displayName} · ${formatContextWindow(model.contextWindow)} ctx`,
        rawId: model.providerModelId,
      }))
      .sort((left, right) =>
        (providerOrder.get(left.providerId) ?? Number.MAX_SAFE_INTEGER) - (providerOrder.get(right.providerId) ?? Number.MAX_SAFE_INTEGER)
        || left.label.localeCompare(right.label)
      )
  }, [dropdownTarget, dropdownQuery, modelConfig, providers])

  const thinkingItems = useMemo((): readonly ThinkingPickerItem[] => {
    if (dropdownTarget?.field !== 'thinking') return []
    const model = effectiveModelForSlot(dropdownTarget.slotId, modelConfig)
    const fallback = DEFAULT_REASONING_EFFORT[dropdownTarget.slotId]
    const efforts = model?.reasoningEfforts.length ? model.reasoningEfforts : [fallback]
    const resolvedDefault = model
      ? resolveReasoningEffort(model, undefined, fallback)
      : fallback
    return [
      {
        id: 'default',
        label: `${formatLabel(resolvedDefault)}${resolvedDefault === 'default' ? '' : ' (default)'}`,
        value: null,
      },
      ...efforts
        .filter((effort) => effort !== resolvedDefault)
        .map((effort) => ({ id: effort, label: formatLabel(effort), value: effort })),
    ]
  }, [dropdownTarget, modelConfig])

  const dropdownItems = dropdownTarget?.field === 'model' ? modelItems : thinkingItems
  const windowStart = Math.max(0, Math.min(dropdownIndex - 2, dropdownItems.length - PICKER_WINDOW_SIZE))
  const visibleDropdownItems = dropdownItems.slice(windowStart, windowStart + PICKER_WINDOW_SIZE)

  const cancelInline = useCallback(() => {
    setEditingProviderId(null)
    setDisconnectProviderId(null)
    setInputValue('')
    setError(null)
  }, [])

  const beginEdit = useCallback((providerId: string) => {
    cancelInline()
    setEditingProviderId(providerId)
  }, [cancelInline])

  const beginDisconnect = useCallback((providerId: string) => {
    cancelInline()
    setDisconnectProviderId(providerId)
  }, [cancelInline])

  const handleSave = useCallback(async () => {
    if (!editingProviderId || submitting) return
    const key = inputValue.trim()
    if (!key) { setError('API key is required'); return }
    setSubmitting(true)
    setError(null)
    try {
      await onSaveProviderApiKey(editingProviderId, key)
      cancelInline()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Failed to save API key')
    } finally {
      setSubmitting(false)
    }
  }, [editingProviderId, inputValue, submitting, onSaveProviderApiKey, cancelInline])

  const handleDisconnect = useCallback(async () => {
    if (!disconnectProviderId || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      await onDisconnectProvider(disconnectProviderId)
      cancelInline()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Failed to disconnect provider')
    } finally {
      setSubmitting(false)
    }
  }, [disconnectProviderId, submitting, onDisconnectProvider, cancelInline])

  const openDropdown = useCallback((target: Exclude<DropdownTarget, null>) => {
    setDropdownTarget(target)
    setDropdownIndex(0)
    setDropdownQuery('')
  }, [])

  const closeDropdown = useCallback(() => {
    setDropdownTarget(null)
    setDropdownIndex(0)
    setDropdownQuery('')
  }, [])

  const selectDropdownItem = useCallback((index: number) => {
    if (!dropdownTarget || !modelConfig) return
    const item = dropdownItems[index]
    if (!item) return
    if (dropdownTarget.field === 'model') {
      const model = item as ModelPickerItem
      void modelConfig.updateSlotModel(dropdownTarget.slotId, model.providerId, model.id)
    } else {
      const effort = item as ThinkingPickerItem
      void modelConfig.updateSlotReasoning(dropdownTarget.slotId, effort.value)
    }
    closeDropdown()
  }, [dropdownTarget, dropdownItems, modelConfig, closeDropdown])

  useKeyboard(useCallback((key: KeyEvent) => {
    if (!isVisible) return
    if (dropdownTarget) {
      if (key.name === 'escape') { key.preventDefault(); closeDropdown(); return }
      if (key.name === 'up') { key.preventDefault(); setDropdownIndex((index) => Math.max(0, index - 1)); return }
      if (key.name === 'down') { key.preventDefault(); setDropdownIndex((index) => Math.min(dropdownItems.length - 1, index + 1)); return }
      if (key.name === 'return' || key.name === 'enter') { key.preventDefault(); selectDropdownItem(dropdownIndex); return }
      return
    }
    if (key.name === 'escape') {
      key.preventDefault()
      if (editingProviderId || disconnectProviderId) cancelInline()
      else onClose()
      return
    }
    if (editingProviderId && (key.name === 'return' || key.name === 'enter') && !key.shift) {
      key.preventDefault()
      void handleSave()
    }
  }, [isVisible, dropdownTarget, dropdownItems.length, dropdownIndex, editingProviderId, disconnectProviderId, closeDropdown, selectDropdownItem, cancelInline, onClose, handleSave]))

  if (!isVisible) return null

  return (
    <box style={{ position: 'relative', flexDirection: 'column', height: '100%' }}>
      <box style={{ flexDirection: 'row', paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}><span attributes={TextAttributes.BOLD}>Settings</span></text>
        <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>Esc to close</span></text>
      </box>
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}><text style={{ fg: theme.border }}>{'─'.repeat(80)}</text></box>

      <scrollbox
        scrollX={false}
        scrollbarOptions={{ visible: false }}
        verticalScrollbarOptions={{ visible: true, trackOptions: { width: 1 } }}
        style={{
          flexGrow: 1,
          rootOptions: { flexGrow: 1, backgroundColor: 'transparent' },
          wrapperOptions: { border: false, backgroundColor: 'transparent' },
          contentOptions: { paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1 },
        }}
      >
        <box style={{ flexDirection: 'column' }}>
          <text style={{ fg: theme.foreground }}><span attributes={TextAttributes.BOLD}>Providers</span></text>
          <box style={{ paddingTop: 1, flexDirection: 'column' }}>
            <ProviderRows
              providers={providers}
              auths={providerAuths}
              editingProviderId={editingProviderId}
              disconnectProviderId={disconnectProviderId}
              inputValue={inputValue}
              submitting={submitting}
              error={error}
              onBeginEdit={beginEdit}
              onBeginDisconnect={beginDisconnect}
              onInputChange={(value) => { setInputValue(value); setError(null) }}
              onSave={() => { void handleSave() }}
              onDisconnect={() => { void handleDisconnect() }}
              onRefresh={(providerId) => { void modelConfig?.refreshModels(providerId) }}
              refreshing={modelConfig?.refreshingModels ?? false}
              onCancel={cancelInline}
            />
          </box>

          <box style={{ paddingTop: 1, paddingBottom: 1 }}><text style={{ fg: theme.border }}>{'─'.repeat(76)}</text></box>
          <box style={{ flexDirection: 'row', paddingBottom: 1 }}>
            <text style={{ fg: theme.foreground, flexGrow: 1 }}><span attributes={TextAttributes.BOLD}>Models</span></text>
            {modelConfig && <Button onClick={() => { void modelConfig.refreshModels() }} onMouseOver={() => setRefreshHovered(true)} onMouseOut={() => setRefreshHovered(false)}><text style={{ fg: refreshHovered ? theme.primary : theme.muted }}>{modelConfig.refreshingModels ? '[Refreshing...]' : '[Refresh]'}</text></Button>}
          </box>

          {SLOT_IDS.map((slotId) => {
            const selectedModel = effectiveModelForSlot(slotId, modelConfig)
            const override = modelConfig?.slotConfig?.[slotId]
            const defaultEffort = DEFAULT_REASONING_EFFORT[slotId]
            const currentEffort = selectedModel
              ? resolveReasoningEffort(selectedModel, override?.reasoningEffort, defaultEffort)
              : defaultEffort
            const modelOpen = dropdownTarget?.slotId === slotId && dropdownTarget.field === 'model'
            const thinkingOpen = dropdownTarget?.slotId === slotId && dropdownTarget.field === 'thinking'
            const providerName = providers?.find((provider) => provider.id === selectedModel?.providerId)?.displayName ?? selectedModel?.providerId

            return (
              <box key={slotId} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                <text style={{ fg: theme.primary }}><span attributes={TextAttributes.BOLD}>{SLOT_DISPLAY_NAMES[slotId]}</span></text>
                <box style={{ flexDirection: 'row' }}>
                  <Button onClick={() => modelOpen ? closeDropdown() : openDropdown({ slotId, field: 'model' })} style={{ borderStyle: 'rounded', borderColor: modelOpen ? theme.primary : theme.border, width: 48, paddingLeft: 1, paddingRight: 1 }}>
                    <text style={{ fg: theme.foreground, overflow: 'hidden' }}>{selectedModel ? `${selectedModel.displayName} · ${providerName}` : 'No model'} </text><text style={{ fg: theme.muted }}>▾</text>
                  </Button>
                  <text> </text>
                  <Button onClick={() => thinkingOpen ? closeDropdown() : openDropdown({ slotId, field: 'thinking' })} style={{ borderStyle: 'rounded', borderColor: thinkingOpen ? theme.primary : theme.border, width: 18, paddingLeft: 1, paddingRight: 1 }}>
                    <text style={{ fg: theme.foreground }}>{formatLabel(currentEffort)} </text><text style={{ fg: theme.muted }}>▾</text>
                  </Button>
                </box>
                <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>{SLOT_DESCRIPTIONS[slotId]}</span></text>

                {(modelOpen || thinkingOpen) && (
                  <box style={{ flexDirection: 'column', borderStyle: 'single', borderColor: theme.primary, width: modelOpen ? 72 : 28, paddingLeft: 1, paddingRight: 1, marginTop: 1 }}>
                    {modelOpen && <SingleLineInput value={dropdownQuery} onChange={(value) => { setDropdownQuery(value); setDropdownIndex(0) }} placeholder="Search models" focused />}
                    {windowStart > 0 && <text style={{ fg: theme.muted }}>↑ {windowStart} more</text>}
                    {visibleDropdownItems.map((item, visibleIndex) => {
                      const index = windowStart + visibleIndex
                      const selected = index === dropdownIndex
                      const modelItem = modelOpen ? item as ModelPickerItem : null
                      const previous = modelOpen && index > 0 ? modelItems[index - 1] : null
                      const showProvider = modelItem && previous?.providerId !== modelItem.providerId
                      return (
                        <box key={modelItem ? JSON.stringify([modelItem.providerId, modelItem.id]) : item.id} style={{ flexDirection: 'column' }}>
                          {showProvider && <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>{modelItem.providerName}</span></text>}
                          <Button onClick={() => selectDropdownItem(index)} onMouseOver={() => setDropdownIndex(index)} style={{ flexDirection: 'column' }}>
                            <text style={{ fg: selected ? theme.primary : theme.foreground }}>{selected ? '▸ ' : '  '}{item.label}</text>
                            {modelItem && <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>  {modelItem.rawId}</span></text>}
                          </Button>
                        </box>
                      )
                    })}
                    {dropdownItems.length === 0 && <text style={{ fg: theme.muted }}>No matching models</text>}
                    {windowStart + PICKER_WINDOW_SIZE < dropdownItems.length && <text style={{ fg: theme.muted }}>↓ {dropdownItems.length - windowStart - PICKER_WINDOW_SIZE} more</text>}
                  </box>
                )}
              </box>
            )
          })}

          {modelConfig && (
            <Button onClick={() => { void modelConfig.resetToDefaults() }} onMouseOver={() => setResetHovered(true)} onMouseOut={() => setResetHovered(false)}>
              <text style={{ fg: resetHovered ? theme.foreground : theme.muted }}>[Reset defaults]</text>
            </Button>
          )}
        </box>
      </scrollbox>
    </box>
  )
})

export type { SettingsOverlayProps }
