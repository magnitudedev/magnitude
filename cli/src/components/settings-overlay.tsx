import { memo, useMemo, useState, useCallback, useEffect, useRef } from 'react'
import { RGBA, TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { SingleLineInput } from './single-line-input'
import { BOX_CHARS } from '../utils/ui-constants'
import { LocalProviderPage } from './local-provider-page'
import type { ProviderDefinition, DetectedProvider, ModelSelection, ProviderAuthMethodStatus, MagnitudeSlot } from '@magnitudedev/agent'
import type { ModelSelectItem } from '../hooks/use-model-select-navigation'
import type { SettingsTab } from '../hooks/use-settings-navigation'
import { SLOT_UI_ORDER } from './setup-wizard-overlay'
import type { Preset, ProviderOptions } from '@magnitudedev/storage'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'github-copilot': '(GitHub.com or Enterprise)',
  lmstudio: '(Local runtime)',
  ollama: '(Local runtime)',
  'llama.cpp': '(Local runtime)',
  'openai-compatible-local': '(DIY OpenAI-compatible local)',
}

function maskApiKey(key: string): string {
  if (key.length <= 8) return '\u2022'.repeat(key.length)
  return key.slice(0, 4) + '\u2022'.repeat(Math.min(key.length - 8, 20)) + key.slice(-4)
}

function getSourceLabel(auth: DetectedProvider['auth']): string {
  if (!auth) return 'No Auth Needed'
  switch (auth.type) {
    case 'oauth': return 'Subscription'
    case 'api': return 'API Key'
    case 'aws': return 'AWS Credentials'
    case 'gcp': return 'GCP Credentials'
    default: return 'Connected'
  }
}

interface SettingsOverlayProps {
  activeTab: SettingsTab
  onTabChange: (tab: SettingsTab) => void
  onClose: () => void
  // Model tab — picker sub-view
  modelItems: ModelSelectItem[]
  modelSelectedIndex: number
  onModelSelect: (providerId: string, modelId: string) => void
  onModelHoverIndex?: (index: number) => void
  modelSearch: string
  onModelSearchChange: (value: string) => void
  showAllProviders: boolean
  onToggleShowAllProviders: () => void
  showRecommendedOnly: boolean
  onToggleShowRecommendedOnly: () => void
  // Provider tab
  allProviders: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  providerSelectedIndex: number
  onProviderSelect: (providerId: string) => void
  onProviderHoverIndex?: (index: number) => void
  // Provider tab — detail view
  providerDetailStatus: ProviderAuthMethodStatus | null
  providerDetailOptions?: ProviderOptions
  providerDetailActions: Array<{ type: string; methodIndex: number; label: string }>
  providerDetailSelectedIndex: number
  onProviderDetailAction: (actionIndex: number) => void
  onProviderDetailHoverIndex?: (index: number) => void
  onLocalProviderSaveEndpoint: (providerId: string, url: string) => void
  onLocalProviderRefreshModels: (providerId: string) => void
  onLocalProviderAddManualModel: (providerId: string, modelId: string) => void
  onLocalProviderRemoveManualModel: (providerId: string, modelId: string) => void
  onLocalProviderSaveOptionalApiKey: (providerId: string, apiKey: string) => void
  // Model tab — per-slot view
  slotModels: Record<MagnitudeSlot, ModelSelection | null>
  selectingModelFor: MagnitudeSlot | null
  onChangeSlot: (slot: MagnitudeSlot) => void
  modelPrefsSelectedIndex: number
  onModelPrefsHoverIndex?: (index: number) => void

  onModelHandleKeyEvent: (key: KeyEvent) => boolean
  onProviderHandleKeyEvent: (key: KeyEvent) => boolean
  onBackFromModelPicker: () => void
  onBackFromProviderDetail: () => void
  presets: Preset[]
  systemDefaultsPresetToken: string
  onSavePreset: (name: string) => void | Promise<void>
  onLoadPreset: (name: string, preferredProviderId?: string) => void | Promise<void>
  onDeletePreset: (name: string) => void | Promise<void>
}

function FilterCheckbox({ label, checked, onToggle }: { label: string; checked: boolean; onToggle: () => void }) {
  const theme = useTheme()
  const [hovered, setHovered] = useState(false)
  const color = checked ? theme.primary : hovered ? theme.foreground : theme.muted
  return (
    <Button onClick={onToggle} onMouseOver={() => setHovered(true)} onMouseOut={() => setHovered(false)}>
      <text style={{ fg: color }}>
        {checked ? '[x]' : '[ ]'} {label}
      </text>
    </Button>
  )
}

function resolveModelDisplay(
  selection: ModelSelection | null,
  modelItems: ModelSelectItem[],
  allProviders: ProviderDefinition[],
): { providerName: string; modelName: string } | null {
  if (!selection) return null
  const item = modelItems.find(m => m.providerId === selection.providerId && m.modelId === selection.modelId)
  if (item) return { providerName: item.providerName, modelName: item.modelName }
  const provider = allProviders.find(p => p.id === selection.providerId)
  const model = provider?.models.find(m => m.id === selection.modelId)
  return {
    providerName: provider?.name ?? selection.providerId,
    modelName: model?.name ?? selection.modelId,
  }
}

export const SettingsOverlay = memo(function SettingsOverlay({
  activeTab,
  onTabChange,
  onClose,
  modelItems,
  modelSelectedIndex,
  onModelSelect,
  onModelHoverIndex,
  modelSearch,
  onModelSearchChange,
  showAllProviders,
  onToggleShowAllProviders,
  showRecommendedOnly,
  onToggleShowRecommendedOnly,
  allProviders,
  detectedProviders,
  providerSelectedIndex,
  onProviderSelect,
  onProviderHoverIndex,
  providerDetailStatus,
  providerDetailOptions,
  providerDetailActions,
  providerDetailSelectedIndex,
  onProviderDetailAction,
  onProviderDetailHoverIndex,
  onLocalProviderSaveEndpoint,
  onLocalProviderRefreshModels,
  onLocalProviderAddManualModel,
  onLocalProviderRemoveManualModel,
  onLocalProviderSaveOptionalApiKey,
  slotModels,
  selectingModelFor,
  onChangeSlot,
  modelPrefsSelectedIndex,
  onModelPrefsHoverIndex,

  onModelHandleKeyEvent,
  onProviderHandleKeyEvent,
  onBackFromModelPicker,
  onBackFromProviderDetail,
  presets,
  systemDefaultsPresetToken,
  onSavePreset,
  onLoadPreset,
  onDeletePreset,
}: SettingsOverlayProps) {
  const theme = useTheme()
  const [hoveredTab, setHoveredTab] = useState<SettingsTab | null>(null)
  const [closeHover, setCloseHover] = useState(false)
  const [showLoadPresetModal, setShowLoadPresetModal] = useState(false)
  const [loadPresetSelectedIndex, setLoadPresetSelectedIndex] = useState(0)
  const [loadPresetHover, setLoadPresetHover] = useState(false)
  const [pendingDeletePresetName, setPendingDeletePresetName] = useState<string | null>(null)
  const [hoveredDeletePresetName, setHoveredDeletePresetName] = useState<string | null>(null)
  const [showSavePresetModal, setShowSavePresetModal] = useState(false)
  const [savePresetName, setSavePresetName] = useState('')
  const [savePresetHover, setSavePresetHover] = useState(false)
  const modelPickerScrollboxRef = useRef<any>(null)
  const modelRowRefs = useRef<Map<number, any>>(new Map())
  const [modelPickerInputMode, setModelPickerInputMode] = useState<'mouse' | 'keyboard'>('mouse')
  const modelPickerLastPointerRef = useRef<{ x: number; y: number } | null>(null)

  // Group model items by provider for section headers
  const modelSections = useMemo(() => {
    const groups: Array<{ providerId: string; providerName: string; connected: boolean; entries: Array<{ item: ModelSelectItem; flatIndex: number }> }> = []
    let currentProvider: string | null = null
    let currentGroup: typeof groups[number] | null = null

    for (let i = 0; i < modelItems.length; i++) {
      const item = modelItems[i]
      if (item.providerId !== currentProvider) {
        currentProvider = item.providerId
        currentGroup = {
          providerId: item.providerId,
          providerName: item.providerName,
          connected: item.connected !== false,
          entries: [],
        }
        groups.push(currentGroup)
      }
      currentGroup!.entries.push({ item, flatIndex: i })
    }
    return groups
  }, [modelItems])

  const slotDisplays = Object.fromEntries(
    SLOT_UI_ORDER.map(({ slot }) => [slot, resolveModelDisplay(slotModels[slot], modelItems, allProviders)])
  ) as Record<MagnitudeSlot, ReturnType<typeof resolveModelDisplay>>

  const loadPresetRows = useMemo(() => {
    const rows: Array<
      | { type: 'label'; label: string }
      | { type: 'provider'; providerId: string; label: string }
      | { type: 'preset'; name: string }
    > = []

    rows.push({ type: 'label', label: 'Provider Defaults' })
    for (const dp of detectedProviders) {
      rows.push({ type: 'provider', providerId: dp.provider.id, label: `${dp.provider.name} defaults` })
    }

    if (presets.length > 0) {
      rows.push({ type: 'label', label: 'User Presets' })
      for (const preset of presets) {
        rows.push({ type: 'preset', name: preset.name })
      }
    }

    return rows
  }, [detectedProviders, presets])

  const loadPresetSelectableIndices = useMemo(() => (
    loadPresetRows.flatMap((row, idx) => (row.type === 'label' ? [] : [idx]))
  ), [loadPresetRows])

  const firstLoadPresetSelectableIndex = loadPresetSelectableIndices[0] ?? 0

  const closeLoadPresetModal = useCallback(() => {
    setShowLoadPresetModal(false)
    setPendingDeletePresetName(null)
    setHoveredDeletePresetName(null)
  }, [])

  const TABS: { id: SettingsTab; label: string }[] = [
    { id: 'provider', label: 'Provider' },
    { id: 'model', label: 'Model' },
  ]

  const readPointerCoords = useCallback((event: unknown): { x: number; y: number } | null => {
    if (!event || typeof event !== 'object') return null
    const record = event as Record<string, unknown>

    const pair = (xKey: string, yKey: string) => {
      const x = record[xKey]
      const y = record[yKey]
      return typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y)
        ? { x, y }
        : null
    }

    return (
      pair('x', 'y') ??
      pair('clientX', 'clientY') ??
      pair('screenX', 'screenY') ??
      pair('col', 'row') ??
      pair('column', 'row')
    )
  }, [])

  const onModelPickerMouseMove = useCallback((event?: unknown) => {
    const coords = readPointerCoords(event)
    if (!coords) return

    const last = modelPickerLastPointerRef.current
    const moved = !last || last.x !== coords.x || last.y !== coords.y
    modelPickerLastPointerRef.current = coords

    if (modelPickerInputMode === 'keyboard' && moved) {
      setModelPickerInputMode('mouse')
    }
  }, [readPointerCoords, modelPickerInputMode])

  useEffect(() => {
    if (!selectingModelFor) {
      setModelPickerInputMode('mouse')
      modelPickerLastPointerRef.current = null
    }
  }, [selectingModelFor])

  useEffect(() => {
    if (!selectingModelFor || modelItems.length === 0) return

    const keepSelectedVisible = (): boolean => {
      const scrollbox = modelPickerScrollboxRef.current
      const contentNode = scrollbox?.content
      const rowNode = modelRowRefs.current.get(modelSelectedIndex)

      if (!scrollbox || !contentNode || !rowNode) return false

      const viewportHeight =
        scrollbox.viewport?.height ??
        scrollbox.viewportHeight ??
        0

      if (!(viewportHeight > 0)) return false

      let rowTop = 0
      let node: any = rowNode
      while (node && node !== contentNode) {
        const yogaNode = node.yogaNode || node.getLayoutNode?.()
        if (yogaNode) {
          rowTop += yogaNode.getComputedTop()
        }
        node = node.parent
      }

      if (node !== contentNode) return false

      const rowYogaNode = rowNode.yogaNode || rowNode.getLayoutNode?.()
      const rowHeight = Math.max(1, rowYogaNode?.getComputedHeight?.() ?? 1)

      const scrollTop = typeof scrollbox.scrollTop === 'number' ? scrollbox.scrollTop : 0
      const visibleTop = scrollTop
      const visibleBottom = scrollTop + viewportHeight
      const rowBottom = rowTop + rowHeight

      let targetTop: number | null = null
      if (rowTop < visibleTop) {
        targetTop = rowTop
      } else if (rowBottom > visibleBottom) {
        targetTop = rowBottom - viewportHeight
      }

      if (targetTop == null) return true

      const clampedTop = Math.max(0, targetTop)
      if (typeof scrollbox.scrollTo === 'function') {
        scrollbox.scrollTo(clampedTop)
      } else {
        scrollbox.scrollTop = clampedTop
      }

      return true
    }

    keepSelectedVisible()
  }, [selectingModelFor, modelSelectedIndex, modelItems])

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option

    if (showSavePresetModal) {
      if (key.name === 'escape') {
        key.preventDefault()
        setShowSavePresetModal(false)
        return
      }
      if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        key.preventDefault()
        const trimmed = savePresetName.trim()
        if (trimmed.length > 0) {
          onSavePreset(trimmed)
          setShowSavePresetModal(false)
          setSavePresetName('')
        }
        return
      }
      return
    }

    if (showLoadPresetModal) {
      key.preventDefault()

      if (pendingDeletePresetName && (key.name === 'y' || ((key.name === 'return' || key.name === 'enter') && plain && !key.shift))) {
        onDeletePreset(pendingDeletePresetName)
        setPendingDeletePresetName(null)
        return
      }

      if (pendingDeletePresetName && (key.name === 'n' || key.name === 'escape')) {
        setPendingDeletePresetName(null)
        return
      }

      if (key.name === 'escape') {
        closeLoadPresetModal()
      } else if ((key.name === 'up' || key.name === 'down') && plain) {
        const delta = key.name === 'up' ? -1 : 1
        const currentPos = loadPresetSelectableIndices.findIndex(i => i === loadPresetSelectedIndex)
        const nextPos = currentPos === -1
          ? (delta > 0 ? 0 : Math.max(0, loadPresetSelectableIndices.length - 1))
          : Math.max(0, Math.min(loadPresetSelectableIndices.length - 1, currentPos + delta))
        const nextIndex = loadPresetSelectableIndices[nextPos]
        if (typeof nextIndex === 'number') setLoadPresetSelectedIndex(nextIndex)
      } else if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        const row = loadPresetRows[loadPresetSelectedIndex]
        if (row?.type === 'provider') {
          onLoadPreset(systemDefaultsPresetToken, row.providerId)
          closeLoadPresetModal()
        } else if (row?.type === 'preset') {
          onLoadPreset(row.name)
          closeLoadPresetModal()
        }
      } else if (plain && (key.name === 'd' || key.name === 'delete' || key.name === 'backspace')) {
        const row = loadPresetRows[loadPresetSelectedIndex]
        if (row?.type === 'preset') {
          setPendingDeletePresetName(row.name)
        }
      }
      return
    }

    if (key.name === 'escape') {
      key.preventDefault()
      if (activeTab === 'provider' && providerDetailStatus) {
        onBackFromProviderDetail()
      } else if (activeTab === 'model' && selectingModelFor) {
        onBackFromModelPicker()
      } else {
        onClose()
      }
      return
    }

    if (key.name === 'left' && plain) {
      key.preventDefault()
      onTabChange('provider')
      return
    }

    if (key.name === 'right' && plain) {
      key.preventDefault()
      onTabChange('model')
      return
    }

    if (activeTab === 'model' && selectingModelFor && plain && (key.name === 'up' || key.name === 'down')) {
      setModelPickerInputMode('keyboard')
    }

    const handled = activeTab === 'model' ? onModelHandleKeyEvent(key) : onProviderHandleKeyEvent(key)
    if (handled) {
      key.preventDefault()
    }
  }, [
    onClose,
    onTabChange,
    activeTab,
    selectingModelFor,
    providerDetailStatus,
    onBackFromModelPicker,
    onBackFromProviderDetail,
    onModelHandleKeyEvent,
    onProviderHandleKeyEvent,
    showLoadPresetModal,
    loadPresetSelectedIndex,
    loadPresetRows,
    loadPresetSelectableIndices,
    showSavePresetModal,
    savePresetName,
    systemDefaultsPresetToken,
    onSavePreset,
    onLoadPreset,
    onDeletePreset,
    pendingDeletePresetName,
    closeLoadPresetModal,
    setModelPickerInputMode,
  ]))

  return (
    <box style={{ flexDirection: 'column', height: '100%', position: 'relative' }}>
      {/* Header */}
      <box style={{
        flexDirection: 'row',
        paddingLeft: 2,
        paddingRight: 2,
        paddingTop: 1,
        paddingBottom: 1,
        flexShrink: 0,
      }}>
        <text style={{ fg: theme.primary, flexGrow: 1 }}>
          <span attributes={TextAttributes.BOLD}>Settings</span>
        </text>
        <box style={{ flexDirection: 'row' }}>
          <Button
            onClick={onClose}
            onMouseOver={() => setCloseHover(true)}
            onMouseOut={() => setCloseHover(false)}
          >
            <text style={{ fg: closeHover ? theme.foreground : theme.muted }} attributes={TextAttributes.UNDERLINE}>Close</text>
          </Button>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>{' '}(Esc)  |  ←/→ tabs  |  ↑/↓ navigate  |  Enter select</span>
          </text>
        </box>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>
          {'─'.repeat(80)}
        </text>
      </box>

      {/* Tabs */}
      <box style={{ flexDirection: 'row', paddingLeft: 1, paddingRight: 1, flexShrink: 0, alignItems: 'center' }}>
        {TABS.map(tab => (
          <Button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            onMouseOver={() => setHoveredTab(tab.id)}
            onMouseOut={() => setHoveredTab(null)}
          >
            <box style={{
              borderStyle: 'single',
              borderColor: activeTab === tab.id ? theme.primary : hoveredTab === tab.id ? theme.foreground : theme.muted,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text
                style={{ fg: activeTab === tab.id ? theme.primary : theme.foreground, wrapMode: 'none' }}
                attributes={activeTab === tab.id ? TextAttributes.BOLD : 0}
              >
                {tab.label}
              </text>
            </box>
          </Button>
        ))}
        {activeTab === 'model' && !selectingModelFor && (
          <box style={{ flexGrow: 1 }} />
        )}
        {activeTab === 'model' && !selectingModelFor && (
          <>
            <Button
              onClick={() => {
                setLoadPresetSelectedIndex(firstLoadPresetSelectableIndex)
                setPendingDeletePresetName(null)
                setHoveredDeletePresetName(null)
                setShowLoadPresetModal(true)
              }}
              onMouseOver={() => setLoadPresetHover(true)}
              onMouseOut={() => setLoadPresetHover(false)}
            >
              <box style={{
                borderStyle: 'single',
                borderColor: loadPresetHover ? theme.foreground : theme.muted,
                customBorderChars: BOX_CHARS,
                paddingLeft: 1,
                paddingRight: 1,
              }}>
                <text style={{ fg: loadPresetHover ? theme.foreground : theme.muted }}>Load preset</text>
              </box>
            </Button>
            <box style={{ width: 1 }} />
            <Button
              onClick={() => setShowSavePresetModal(true)}
              onMouseOver={() => setSavePresetHover(true)}
              onMouseOut={() => setSavePresetHover(false)}
            >
              <box style={{
                borderStyle: 'single',
                borderColor: savePresetHover ? theme.foreground : theme.muted,
                customBorderChars: BOX_CHARS,
                paddingLeft: 1,
                paddingRight: 1,
              }}>
                <text style={{ fg: savePresetHover ? theme.foreground : theme.muted }}>Save current as preset</text>
              </box>
            </Button>
          </>
        )}
      </box>

      {/* Content area */}
      {activeTab === 'model' ? (
        selectingModelFor ? (
          <>
            {/* Model picker sub-view */}
            <scrollbox
              ref={modelPickerScrollboxRef}
              onMouseMove={onModelPickerMouseMove}
              scrollX={false}
              scrollbarOptions={{ visible: false }}
              verticalScrollbarOptions={{
                visible: true,
                trackOptions: { width: 1 },
              }}
              style={{
                flexGrow: 1,
                rootOptions: {
                  flexGrow: 1,
                  backgroundColor: 'transparent',
                },
                wrapperOptions: {
                  border: false,
                  backgroundColor: 'transparent',
                },
                contentOptions: {
                  paddingLeft: 1,
                  paddingRight: 1,
                  paddingTop: 1,
                },
              }}
            >
              <box style={{ paddingLeft: 1, paddingBottom: 1, flexDirection: 'column' }}>
                <text style={{ fg: theme.warning }}>
                  Select a model for your <span attributes={TextAttributes.BOLD}>{SLOT_UI_ORDER.find(s => s.slot === selectingModelFor)?.label ?? selectingModelFor}</span> model. Press Esc to go back.
                </text>
                <box style={{ paddingTop: 1, paddingRight: 1, flexDirection: 'column' }}>
                  <box style={{
                    borderStyle: 'single',
                    borderColor: theme.border,
                    customBorderChars: BOX_CHARS,
                    paddingLeft: 1,
                  }}>
                    <SingleLineInput
                      value={modelSearch}
                      onChange={onModelSearchChange}
                      placeholder="Search providers or models"
                    />
                  </box>
                  <box style={{ flexDirection: 'row', gap: 2 }}>
                    <FilterCheckbox label="Connected only" checked={!showAllProviders} onToggle={onToggleShowAllProviders} />
                    <FilterCheckbox label="Recommended only" checked={showRecommendedOnly} onToggle={onToggleShowRecommendedOnly} />
                  </box>
                </box>
              </box>
              {modelItems.length === 0 ? (
                <box style={{ paddingLeft: 1 }}>
                  <text style={{ fg: theme.muted }}>No models available.</text>
                </box>
              ) : modelSections.length === 0 ? (
                <box style={{ paddingLeft: 1 }}>
                  <text style={{ fg: theme.muted }}>No search matches.</text>
                </box>
              ) : (
                modelSections.map((section) => (
                  <box key={section.providerId} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                    <box style={{ paddingLeft: 1, paddingBottom: 0 }}>
                      <text style={{ fg: theme.muted }}>
                        <span attributes={TextAttributes.BOLD}>{section.providerName}</span>
                        {!section.connected && <span attributes={TextAttributes.DIM}> — not connected</span>}
                      </text>
                    </box>
                    {section.entries.map(({ item, flatIndex }) => {
                      const isSelected = flatIndex === modelSelectedIndex
                      const selectable = item.selectable !== false

                      return (
                        <box
                          key={`${item.providerId}:${item.modelId}`}
                          ref={(node: any) => {
                            if (node) modelRowRefs.current.set(flatIndex, node)
                            else modelRowRefs.current.delete(flatIndex)
                          }}
                        >
                          <Button
                            onClick={() => {
                              setModelPickerInputMode('mouse')
                              if (selectable) onModelSelect(item.providerId, item.modelId)
                            }}
                            onMouseOver={() => {
                              if (selectable && modelPickerInputMode === 'mouse') onModelHoverIndex?.(flatIndex)
                            }}
                            style={{
                              flexDirection: 'column',
                              paddingLeft: 1,
                              paddingRight: 1,
                              backgroundColor: isSelected ? theme.surface : undefined,
                            }}
                          >
                            <text style={{ fg: !selectable ? theme.muted : isSelected ? theme.primary : theme.foreground, flexGrow: 1 }}>
                              {isSelected ? '> ' : '  '}
                              <span style={{ fg: item.recommended ? theme.primary : undefined }}>
                                {item.recommended ? '[*] ' : '    '}
                              </span>
                              {item.modelName}
                              <span attributes={TextAttributes.DIM}> — {item.modelId}</span>
                            </text>
                          </Button>
                        </box>
                      )
                    })}
                  </box>
                ))
              )}
            </scrollbox>

            {/* Model picker footer */}
            <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>
                  {modelItems.length} model{modelItems.length === 1 ? '' : 's'} across {modelSections.length} provider{modelSections.length === 1 ? '' : 's'}
                </span>
              </text>
            </box>
          </>
        ) : (
          <>
            {/* Primary/Secondary model view */}
            <scrollbox
              scrollX={false}
              scrollbarOptions={{ visible: false }}
              verticalScrollbarOptions={{
                visible: true,
                trackOptions: { width: 1 },
              }}
              style={{
                flexGrow: 1,
                rootOptions: {
                  flexGrow: 1,
                  backgroundColor: 'transparent',
                },
                wrapperOptions: {
                  border: false,
                  backgroundColor: 'transparent',
                },
                contentOptions: {
                  paddingLeft: 2,
                  paddingRight: 2,
                  paddingTop: 1,
                },
              }}
            >
              {SLOT_UI_ORDER.map(({ slot, label, description }, idx) => {
                const display = slotDisplays[slot]
                return (
                  <box key={slot} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                    <box style={{ paddingBottom: 0 }}>
                      <text style={{ fg: theme.foreground }}>
                        <span attributes={TextAttributes.BOLD}>{label}</span> {description}
                      </text>
                    </box>
                    <Button
                      onClick={() => onChangeSlot(slot)}
                      onMouseOver={() => onModelPrefsHoverIndex?.(idx)}
                    >
                      <box style={{
                        flexDirection: 'row',
                        borderStyle: 'single',
                        borderColor: modelPrefsSelectedIndex === idx ? theme.primary : theme.border,
                        customBorderChars: BOX_CHARS,
                        paddingLeft: 1,
                        paddingRight: 1,
                      }}>
                        <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                          {display ? (
                            <>
                              {display.providerName}
                              <span attributes={TextAttributes.DIM}> · </span>
                              {display.modelName}
                            </>
                          ) : (
                            <span style={{ fg: theme.muted }}>Not configured</span>
                          )}
                        </text>
                        <text style={{ fg: modelPrefsSelectedIndex === idx ? theme.primary : theme.muted }}>
                          {' [Change]'}
                        </text>
                      </box>
                    </Button>
                  </box>
                )
              })}
            </scrollbox>

            {/* Model prefs footer */}
            <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>
                  Press Enter or click [Change] to select a model
                </span>
              </text>
            </box>
          </>
        )
      ) : providerDetailStatus ? (() => {
        // Build map of methodIndex -> actions with their global indices
        const methodActionMap = new Map<number, Array<{ globalIdx: number; action: typeof providerDetailActions[0] }>>()
        providerDetailActions.forEach((action, globalIdx) => {
          const existing = methodActionMap.get(action.methodIndex) ?? []
          existing.push({ globalIdx, action })
          methodActionMap.set(action.methodIndex, existing)
        })

        const provider = providerDetailStatus.provider
        const isLocalProvider = provider.providerFamily === 'local'
        const baseUrl = typeof providerDetailOptions?.baseUrl === 'string' ? providerDetailOptions.baseUrl.trim() : ''
        const hasBaseUrl = baseUrl.length > 0
        const discoveredModels = Array.isArray(providerDetailOptions?.discoveredModels)
          ? providerDetailOptions.discoveredModels
            .filter((m): m is { id: string; name?: string } => typeof m?.id === 'string')
            .map((m) => ({ id: m.id, name: typeof m.name === 'string' && m.name.trim().length > 0 ? m.name : m.id }))
          : []
        const discoveredCount = discoveredModels.length
        const rememberedModelIdsRaw = Array.isArray(providerDetailOptions?.rememberedModelIds)
          ? providerDetailOptions.rememberedModelIds.filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
          : []
        const discoveredModelIdSet = new Set(discoveredModels.map((m) => m.id))
        const rememberedModelIds = rememberedModelIdsRaw.filter((id) => !discoveredModelIdSet.has(id))
        const lastDiscoveryError = typeof providerDetailOptions?.lastDiscoveryError === 'string' && providerDetailOptions.lastDiscoveryError.trim().length > 0
          ? providerDetailOptions.lastDiscoveryError
          : null
        const optionalApiKeyMethod = providerDetailStatus.methods.find((methodStatus) => methodStatus.method.type === 'api-key')
        const optionalApiKeyValue = optionalApiKeyMethod?.auth?.type === 'api' ? optionalApiKeyMethod.auth.key : ''
        const hasSavedOptionalApiKey = optionalApiKeyMethod?.auth?.type === 'api'

        return (
          <>
            <scrollbox
              scrollX={false}
              scrollbarOptions={{ visible: false }}
              verticalScrollbarOptions={{
                visible: true,
                trackOptions: { width: 1 },
              }}
              style={{
                flexGrow: 1,
                rootOptions: {
                  flexGrow: 1,
                  backgroundColor: 'transparent',
                },
                wrapperOptions: {
                  border: false,
                  backgroundColor: 'transparent',
                },
                contentOptions: {
                  paddingLeft: 2,
                  paddingRight: 2,
                  paddingTop: 1,
                },
              }}
            >
              <box style={{ flexDirection: 'column' }}>
                {/* Provider name */}
                <box style={{ paddingBottom: 1 }}>
                  <text style={{ fg: theme.primary }}>
                    <span attributes={TextAttributes.BOLD}>{provider.name}</span>
                  </text>
                </box>

                {isLocalProvider && (
                  <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
                    <LocalProviderPage
                      providerName={provider.name}
                      showProviderTitle={false}
                      endpoint={baseUrl}
                      endpointPlaceholder={provider.defaultBaseUrl ?? 'http://localhost:1234/v1'}
                      discoveredModels={discoveredModels}
                      manualModelIds={rememberedModelIds}
                      lastDiscoveryError={lastDiscoveryError}
                      onSaveEndpoint={(url) => onLocalProviderSaveEndpoint(provider.id, url)}
                      onRefreshModels={() => onLocalProviderRefreshModels(provider.id)}
                      onAddManualModel={(modelId) => onLocalProviderAddManualModel(provider.id, modelId)}
                      onRemoveManualModel={(modelId) => onLocalProviderRemoveManualModel(provider.id, modelId)}
                      showOptionalApiKey={!!optionalApiKeyMethod}
                      optionalApiKeyValue={optionalApiKeyValue}
                      hasSavedOptionalApiKey={hasSavedOptionalApiKey}
                      onSaveOptionalApiKey={(apiKey) => onLocalProviderSaveOptionalApiKey(provider.id, apiKey)}
                    />
                  </box>
                )}

                {/* Auth methods */}
              {providerDetailStatus.methods
                .filter((m) => !(isLocalProvider && m.method.type === 'api-key'))
                .map((m) => {
                  const isApiKey = m.method.type === 'api-key'
                  const apiKeyValue = isApiKey && m.auth?.type === 'api' ? (m.auth as { type: 'api'; key: string }).key : null
                  const methodActions = methodActionMap.get(m.methodIndex) ?? []

                  return (
                    <box key={m.methodIndex} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                      {/* Method status line */}
                      <box>
                        <text style={{ fg: theme.foreground }}>
                          {'  '}
                          {m.connected ? (
                            <span style={{ fg: theme.success }}>{'\u2713 '}</span>
                          ) : (
                            <span style={{ fg: theme.muted }}>{'\u00b7 '}</span>
                          )}
                          {m.method.label}
                          {m.connected && (
                            <span style={{ fg: theme.success }}>
                              {' \u2014 Connected'}
                              {m.source === 'env' ? ' (from environment)' : ''}
                            </span>
                          )}
                        </text>
                      </box>

                      {/* Masked API key if applicable */}
                      {apiKeyValue && (
                        <box style={{ paddingLeft: 4 }}>
                          <text style={{ fg: theme.foreground }}>
                            {'Key: '}
                            <span attributes={TextAttributes.DIM}>{maskApiKey(apiKeyValue)}</span>
                          </text>
                        </box>
                      )}

                      {/* Action buttons for this method */}
                      {methodActions.map(({ globalIdx, action }) => (
                        <Button
                          key={globalIdx}
                          onClick={() => onProviderDetailAction(globalIdx)}
                          onMouseOver={() => onProviderDetailHoverIndex?.(globalIdx)}
                          style={{
                            paddingLeft: 3,
                            paddingRight: 1,
                            backgroundColor: providerDetailSelectedIndex === globalIdx ? theme.surface : undefined,
                          }}
                        >
                          <text style={{
                            fg: action.type === 'disconnect'
                              ? (providerDetailSelectedIndex === globalIdx ? theme.error : theme.foreground)
                              : (providerDetailSelectedIndex === globalIdx ? theme.primary : theme.foreground)
                          }}>
                            {providerDetailSelectedIndex === globalIdx ? '> ' : '  '}
                            {`[${action.label}]`}
                          </text>
                        </Button>
                      ))}
                    </box>
                  )
                })}


              </box>
            </scrollbox>

            {/* Detail footer */}
            <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>Press Esc to go back</span>
              </text>
            </box>
          </>
        )
      })() : (
        <>
          {/* Provider list */}
          <scrollbox
            scrollX={false}
            scrollbarOptions={{ visible: false }}
            verticalScrollbarOptions={{
              visible: true,
              trackOptions: { width: 1 },
            }}
            style={{
              flexGrow: 1,
              rootOptions: {
                flexGrow: 1,
                backgroundColor: 'transparent',
              },
              wrapperOptions: {
                border: false,
                backgroundColor: 'transparent',
              },
              contentOptions: {
                paddingLeft: 1,
                paddingRight: 1,
                paddingTop: 1,
              },
            }}
          >
            {allProviders.map((provider, index) => {
              const detected = detectedProviders.find(d => d.provider.id === provider.id)
              const isConnected = !!detected
              const isSelected = index === providerSelectedIndex
              const description = PROVIDER_DESCRIPTIONS[provider.id]
              const sourceLabel = isConnected ? getSourceLabel(detected!.auth) : null

              return (
                <Button
                  key={provider.id}
                  onClick={() => onProviderSelect(provider.id)}
                  onMouseOver={() => onProviderHoverIndex?.(index)}
                  style={{
                    flexDirection: 'row',
                    paddingLeft: 1,
                    paddingRight: 1,
                    backgroundColor: isSelected ? theme.surface : undefined,
                  }}
                >
                  <text style={{ fg: isSelected ? theme.primary : theme.foreground }}>
                    {isSelected ? '> ' : '  '}
                    {isConnected ? (
                      <span style={{ fg: theme.success }}>{'✓ '}</span>
                    ) : (
                      <span style={{ fg: theme.muted }}>{'· '}</span>
                    )}
                    {provider.name}
                    {description && (
                      <span attributes={TextAttributes.DIM}>{' '}{description}</span>
                    )}
                    {isConnected && (
                      <span attributes={TextAttributes.DIM}> — Connected ({sourceLabel})</span>
                    )}
                  </text>
                </Button>
              )
            })}
          </scrollbox>

          {/* Provider footer */}
          <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
            <text style={{ fg: theme.muted }}>
              <span attributes={TextAttributes.DIM}>
                {allProviders.length} provider{allProviders.length === 1 ? '' : 's'}
                {' · '}
                {detectedProviders.length} connected
              </span>
            </text>
          </box>
        </>
      )}

      {showLoadPresetModal && (
        <box style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 20,
          alignItems: 'center', justifyContent: 'center', backgroundColor: RGBA.fromInts(0, 0, 0, 153),
        }}>
          <box style={{
            borderStyle: 'single', border: ['left', 'right', 'top', 'bottom'], borderColor: theme.border,
            backgroundColor: theme.surface, customBorderChars: BOX_CHARS, minWidth: 52, maxWidth: 72,
          }}>
            <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column' }}>
              <box style={{ paddingBottom: 1 }}>
                <text style={{ fg: theme.primary }}><span attributes={TextAttributes.BOLD}>Load preset</span></text>
              </box>
              {loadPresetRows.map((row, rowIndex) => {
                if (row.type === 'label') {
                  return (
                    <box key={`label:${row.label}`} style={{ paddingLeft: 1, paddingTop: rowIndex === 0 ? 0 : 1 }}>
                      <text style={{ fg: theme.muted }}><span attributes={TextAttributes.BOLD}>{row.label}</span></text>
                    </box>
                  )
                }

                const isSelected = loadPresetSelectedIndex === rowIndex
                if (row.type === 'provider') {
                  return (
                    <Button
                      key={`provider:${row.providerId}`}
                      onClick={() => { onLoadPreset(systemDefaultsPresetToken, row.providerId); closeLoadPresetModal() }}
                      onMouseOver={() => setLoadPresetSelectedIndex(rowIndex)}
                      style={{ paddingLeft: 1, paddingRight: 1, backgroundColor: isSelected ? theme.background : undefined }}
                    >
                      <text style={{ fg: isSelected ? theme.primary : theme.foreground }}>
                        {isSelected ? '> ' : '  '}{row.label}
                      </text>
                    </Button>
                  )
                }

                const isPendingDelete = pendingDeletePresetName === row.name
                const isDeleteHovered = hoveredDeletePresetName === row.name
                return (
                  <box key={`preset:${row.name}`} style={{ flexDirection: 'row' }}>
                    <Button
                      onClick={() => {
                        if (isPendingDelete) return
                        onLoadPreset(row.name)
                        closeLoadPresetModal()
                      }}
                      onMouseOver={() => setLoadPresetSelectedIndex(rowIndex)}
                      style={{ flexGrow: 1, paddingLeft: 1, paddingRight: 1, backgroundColor: isSelected ? theme.background : undefined }}
                    >
                      <text style={{ fg: isSelected ? theme.primary : theme.foreground }}>
                        {isSelected ? '> ' : '  '}
                        {isPendingDelete ? `Delete ${row.name}? [Y/n]` : row.name}
                      </text>
                    </Button>
                    {!isPendingDelete && (
                      <Button
                        onClick={() => {
                          setLoadPresetSelectedIndex(rowIndex)
                          setPendingDeletePresetName(row.name)
                        }}
                        onMouseOver={() => {
                          setLoadPresetSelectedIndex(rowIndex)
                          setHoveredDeletePresetName(row.name)
                        }}
                        onMouseOut={() => setHoveredDeletePresetName(current => current === row.name ? null : current)}
                        style={{ paddingRight: 1, backgroundColor: isDeleteHovered ? theme.background : undefined }}
                      >
                        <text
                          style={{ fg: isDeleteHovered ? theme.error : theme.muted }}
                          attributes={isDeleteHovered ? TextAttributes.BOLD : 0}
                        >
                          [Delete]
                        </text>
                      </Button>
                    )}
                  </box>
                )
              })}
              <box style={{ paddingTop: 1 }}>
                <text style={{ fg: theme.muted }}>
                  <span attributes={TextAttributes.DIM}>
                    {pendingDeletePresetName
                      ? 'Y/Enter confirm delete  |  N/Esc cancel'
                      : '↑/↓ navigate  |  Enter load  |  D delete  |  Esc cancel'}
                  </span>
                </text>
              </box>
            </box>
          </box>
        </box>
      )}

      {showSavePresetModal && (
        <box style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 20,
          alignItems: 'center', justifyContent: 'center', backgroundColor: RGBA.fromInts(0, 0, 0, 153),
        }}>
          <box style={{
            borderStyle: 'single', border: ['left', 'right', 'top', 'bottom'], borderColor: theme.border,
            backgroundColor: theme.surface, customBorderChars: BOX_CHARS, minWidth: 52, maxWidth: 72,
          }}>
            <box style={{ paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1, flexDirection: 'column' }}>
              <box style={{ paddingBottom: 1 }}>
                <text style={{ fg: theme.primary }}><span attributes={TextAttributes.BOLD}>Save current as preset</span></text>
              </box>
              <box style={{ borderStyle: 'single', borderColor: theme.border, customBorderChars: BOX_CHARS, paddingLeft: 1 }}>
                <SingleLineInput value={savePresetName} onChange={setSavePresetName} placeholder="Preset name" />
              </box>
              <box style={{ paddingTop: 1 }}>
                <Button onClick={() => {
                  const trimmed = savePresetName.trim()
                  if (!trimmed) return
                  onSavePreset(trimmed)
                  setShowSavePresetModal(false)
                  setSavePresetName('')
                }}>
                  <text style={{ fg: theme.primary }}>[Save]</text>
                </Button>
                <text style={{ fg: theme.muted }}><span attributes={TextAttributes.DIM}>{'  Enter save  |  Esc cancel'}</span></text>
              </box>
            </box>
          </box>
        </box>
      )}
    </box>
  )
})
