import { memo, useMemo, useState, useCallback } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { BOX_CHARS } from '../utils/ui-constants'
import type { ProviderDefinition, DetectedProvider, ModelSelection, ProviderAuthMethodStatus } from '@magnitudedev/agent'
import type { ModelSelectItem } from '../hooks/use-model-select-navigation'
import type { SettingsTab } from '../hooks/use-settings-navigation'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'github-copilot': '(GitHub.com or Enterprise)',
  local: '(Ollama, LM Studio, llama.cpp, vLLM)',
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
  modelProviders: ProviderDefinition[]
  modelItems: ModelSelectItem[]
  modelSelectedIndex: number
  onModelSelect: (providerId: string, modelId: string) => void
  onModelHoverIndex?: (index: number) => void
  // Provider tab
  allProviders: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  providerSelectedIndex: number
  onProviderSelect: (providerId: string) => void
  onProviderHoverIndex?: (index: number) => void
  // Provider tab — detail view
  providerDetailStatus: ProviderAuthMethodStatus | null
  providerDetailActions: Array<{ type: string; methodIndex: number; label: string }>
  providerDetailSelectedIndex: number
  onProviderDetailAction: (actionIndex: number) => void
  onProviderDetailHoverIndex?: (index: number) => void
  // Model tab — primary/secondary/browser view
  primaryModel: ModelSelection | null
  secondaryModel: ModelSelection | null
  browserModel: ModelSelection | null
  selectingModelFor: 'primary' | 'secondary' | 'browser' | null
  onChangePrimary: () => void
  onChangeSecondary: () => void
  onChangeBrowser: () => void
  modelPrefsSelectedIndex: number
  onModelPrefsHoverIndex?: (index: number) => void
  localProviderConfig?: { baseUrl?: string | null; modelId?: string | null } | null
  localProviderAuth?: { type: 'api'; key: string } | null
  onModelHandleKeyEvent: (key: KeyEvent) => boolean
  onProviderHandleKeyEvent: (key: KeyEvent) => boolean
  onBackFromModelPicker: () => void
  onBackFromProviderDetail: () => void
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
  return {
    providerName: provider?.name ?? selection.providerId,
    modelName: selection.modelId,
  }
}

export const SettingsOverlay = memo(function SettingsOverlay({
  activeTab,
  onTabChange,
  onClose,
  modelProviders,
  modelItems,
  modelSelectedIndex,
  onModelSelect,
  onModelHoverIndex,
  allProviders,
  detectedProviders,
  providerSelectedIndex,
  onProviderSelect,
  onProviderHoverIndex,
  providerDetailStatus,
  providerDetailActions,
  providerDetailSelectedIndex,
  onProviderDetailAction,
  onProviderDetailHoverIndex,
  primaryModel,
  secondaryModel,
  browserModel,
  selectingModelFor,
  onChangePrimary,
  onChangeSecondary,
  onChangeBrowser,
  modelPrefsSelectedIndex,
  onModelPrefsHoverIndex,
  localProviderConfig,
  localProviderAuth,
  onModelHandleKeyEvent,
  onProviderHandleKeyEvent,
  onBackFromModelPicker,
  onBackFromProviderDetail,
}: SettingsOverlayProps) {
  const theme = useTheme()
  const [hoveredTab, setHoveredTab] = useState<SettingsTab | null>(null)
  const [closeHover, setCloseHover] = useState(false)

  // Group model items by provider for section headers
  const modelSections = useMemo(() => {
    const groups: Array<{ providerName: string; entries: Array<{ item: ModelSelectItem; flatIndex: number }> }> = []
    let currentProvider: string | null = null
    let currentGroup: typeof groups[number] | null = null

    for (let i = 0; i < modelItems.length; i++) {
      const item = modelItems[i]
      if (item.providerId !== currentProvider) {
        currentProvider = item.providerId
        currentGroup = { providerName: item.providerName, entries: [] }
        groups.push(currentGroup)
      }
      currentGroup!.entries.push({ item, flatIndex: i })
    }
    return groups
  }, [modelItems])

  const primaryDisplay = resolveModelDisplay(primaryModel, modelItems, allProviders)
  const secondaryDisplay = resolveModelDisplay(secondaryModel, modelItems, allProviders)
  const browserDisplay = resolveModelDisplay(browserModel, modelItems, allProviders)

  const TABS: { id: SettingsTab; label: string }[] = [
    { id: 'provider', label: 'Provider' },
    { id: 'model', label: 'Model' },
  ]

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option

    if (key.name === 'escape') {
      key.preventDefault()
      onClose()
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

    if (activeTab === 'model' && selectingModelFor && key.name === 'b' && plain && !key.shift) {
      key.preventDefault()
      onBackFromModelPicker()
      return
    }

    if (activeTab === 'provider' && providerDetailStatus && key.name === 'b' && plain && !key.shift) {
      key.preventDefault()
      onBackFromProviderDetail()
      return
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
  ]))

  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
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
      <box style={{ flexDirection: 'row', paddingLeft: 1, flexShrink: 0 }}>
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
      </box>

      {/* Content area */}
      {activeTab === 'model' ? (
        selectingModelFor ? (
          <>
            {/* Model picker sub-view */}
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
              <box style={{ paddingLeft: 1, paddingBottom: 1 }}>
                <text style={{ fg: theme.warning }}>
                  Select a model for your <span attributes={TextAttributes.BOLD}>{selectingModelFor === 'primary' ? 'Primary' : selectingModelFor === 'secondary' ? 'Secondary' : 'Browser Agent'}</span> model. Press b to go back.
                </text>
              </box>
              {modelSections.length === 0 ? (
                <box style={{ paddingLeft: 1 }}>
                  <text style={{ fg: theme.muted }}>No models available.</text>
                </box>
              ) : (
                modelSections.map((section) => (
                  <box key={section.providerName} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                    <box style={{ paddingLeft: 1, paddingBottom: 0 }}>
                      <text style={{ fg: theme.muted }}>
                        <span attributes={TextAttributes.BOLD}>{section.providerName}</span>
                      </text>
                    </box>
                    {section.entries.map(({ item, flatIndex }) => {
                      const isSelected = flatIndex === modelSelectedIndex
                      return (
                        <Button
                          key={`${item.providerId}:${item.modelId}`}
                          onClick={() => onModelSelect(item.providerId, item.modelId)}
                          onMouseOver={() => onModelHoverIndex?.(flatIndex)}
                          style={{
                            flexDirection: 'row',
                            paddingLeft: 1,
                            paddingRight: 1,
                            backgroundColor: isSelected ? theme.surface : undefined,
                          }}
                        >
                          <text style={{ fg: isSelected ? theme.primary : theme.foreground, flexGrow: 1 }}>
                            {isSelected ? '> ' : '  '}{item.modelName}
                            <span attributes={TextAttributes.DIM}> — {item.modelId}</span>
                          </text>
                        </Button>
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
            <box style={{
              flexDirection: 'column',
              paddingLeft: 2,
              paddingRight: 2,
              paddingTop: 1,
              flexGrow: 1,
            }}>
              {/* Primary Model */}
              <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
                <box style={{ paddingBottom: 0 }}>
                  <text style={{ fg: theme.foreground }}>
                    <span attributes={TextAttributes.BOLD}>Primary Model</span>
                    <span attributes={TextAttributes.DIM}> (Smarter)</span>
                  </text>
                </box>
                <Button
                  onClick={onChangePrimary}
                  onMouseOver={() => onModelPrefsHoverIndex?.(0)}
                >
                  <box style={{
                    flexDirection: 'row',
                    borderStyle: 'single',
                    borderColor: modelPrefsSelectedIndex === 0 ? theme.primary : theme.border,
                    customBorderChars: BOX_CHARS,
                    paddingLeft: 1,
                    paddingRight: 1,
                  }}>
                    <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                      {primaryDisplay ? (
                        <>
                          {primaryDisplay.providerName}
                          <span attributes={TextAttributes.DIM}> · </span>
                          {primaryDisplay.modelName}
                        </>
                      ) : (
                        <span style={{ fg: theme.muted }}>Not configured</span>
                      )}
                    </text>
                    <text style={{ fg: modelPrefsSelectedIndex === 0 ? theme.primary : theme.muted }}>
                      {' [Change]'}
                    </text>
                  </box>
                </Button>
              </box>

              {/* Secondary Model */}
              <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
                <box style={{ paddingBottom: 0 }}>
                  <text style={{ fg: theme.foreground }}>
                    <span attributes={TextAttributes.BOLD}>Secondary Model</span>
                    <span attributes={TextAttributes.DIM}> (Faster)</span>
                  </text>
                </box>
                <Button
                  onClick={onChangeSecondary}
                  onMouseOver={() => onModelPrefsHoverIndex?.(1)}
                >
                  <box style={{
                    flexDirection: 'row',
                    borderStyle: 'single',
                    borderColor: modelPrefsSelectedIndex === 1 ? theme.primary : theme.border,
                    customBorderChars: BOX_CHARS,
                    paddingLeft: 1,
                    paddingRight: 1,
                  }}>
                    <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                      {secondaryDisplay ? (
                        <>
                          {secondaryDisplay.providerName}
                          <span attributes={TextAttributes.DIM}> · </span>
                          {secondaryDisplay.modelName}
                        </>
                      ) : (
                        <span style={{ fg: theme.muted }}>Not configured</span>
                      )}
                    </text>
                    <text style={{ fg: modelPrefsSelectedIndex === 1 ? theme.primary : theme.muted }}>
                      {' [Change]'}
                    </text>
                  </box>
                </Button>
              </box>

              {/* Browser Agent Model */}
              <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
                <box style={{ paddingBottom: 0 }}>
                  <text style={{ fg: theme.foreground }}>
                    <span attributes={TextAttributes.BOLD}>Browser Agent Model</span>
                    <span attributes={TextAttributes.DIM}> (Vision)</span>
                  </text>
                </box>
                <Button
                  onClick={onChangeBrowser}
                  onMouseOver={() => onModelPrefsHoverIndex?.(2)}
                >
                  <box style={{
                    flexDirection: 'row',
                    borderStyle: 'single',
                    borderColor: modelPrefsSelectedIndex === 2 ? theme.primary : theme.border,
                    customBorderChars: BOX_CHARS,
                    paddingLeft: 1,
                    paddingRight: 1,
                  }}>
                    <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                      {browserDisplay ? (
                        <>
                          {browserDisplay.providerName}
                          <span attributes={TextAttributes.DIM}> · </span>
                          {browserDisplay.modelName}
                        </>
                      ) : (
                        <span style={{ fg: theme.muted }}>Not configured</span>
                      )}
                    </text>
                    <text style={{ fg: modelPrefsSelectedIndex === 2 ? theme.primary : theme.muted }}>
                      {' [Change]'}
                    </text>
                  </box>
                </Button>
              </box>
            </box>

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

        return (
          <>
            <box style={{ flexDirection: 'column', paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1 }}>
              {/* Provider name */}
              <box style={{ paddingBottom: 1 }}>
                <text style={{ fg: theme.primary }}>
                  <span attributes={TextAttributes.BOLD}>{providerDetailStatus.provider.name}</span>
                </text>
              </box>

              {/* Auth methods */}
              {providerDetailStatus.methods.map((m) => {
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

                    {/* Local provider details */}
                    {m.connected && m.method.type === 'none' && providerDetailStatus.provider.id === 'local' && (() => {
                      return (
                        <>
                          {localProviderConfig?.baseUrl && (
                            <box style={{ paddingLeft: 4 }}>
                              <text style={{ fg: theme.foreground }}>
                                {'URL: '}
                                <span attributes={TextAttributes.DIM}>{localProviderConfig.baseUrl}</span>
                              </text>
                            </box>
                          )}
                          {localProviderConfig?.modelId && (
                            <box style={{ paddingLeft: 4 }}>
                              <text style={{ fg: theme.foreground }}>
                                {'Model: '}
                                <span attributes={TextAttributes.DIM}>{localProviderConfig.modelId}</span>
                              </text>
                            </box>
                          )}
                          {localProviderAuth?.type === 'api' && (
                            <box style={{ paddingLeft: 4 }}>
                              <text style={{ fg: theme.foreground }}>
                                {'Key: '}
                                <span attributes={TextAttributes.DIM}>{maskApiKey(localProviderAuth.key)}</span>
                              </text>
                            </box>
                          )}
                        </>
                      )
                    })()}

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

            {/* Detail footer */}
            <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
              <text style={{ fg: theme.muted }}>
                <span attributes={TextAttributes.DIM}>Press b or Esc to go back</span>
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
    </box>
  )
})
