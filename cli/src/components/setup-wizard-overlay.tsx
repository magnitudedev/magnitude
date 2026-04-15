import { memo, useCallback, useMemo, useState } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { slate } from '../utils/theme'
import { Button } from './button'
import { WizardHeader } from './wizard-header'
import { LocalProviderPage } from './local-provider-page'
import { BOX_CHARS } from '../utils/ui-constants'
import type { ProviderDefinition, ModelSelection } from '@magnitudedev/agent'
import type { DetectedProvider } from '@magnitudedev/agent'
import type { MagnitudeSlot } from '@magnitudedev/agent'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'github-copilot': '(GitHub.com or Enterprise)',
  'openai-compatible-local': '(DIY OpenAI-compatible local)',
}

export const SLOT_UI_ORDER: { slot: MagnitudeSlot; label: string; description: string }[] = [
  { slot: 'lead', label: 'Lead', description: '(Coordinates all worker usage)' },
  { slot: 'worker', label: 'Worker', description: '(Implements tasks with skill activation)' },
]

export type WizardStep = 'provider' | 'local-provider' | 'models' | 'browser'

interface SetupWizardOverlayProps {
  step: WizardStep
  hasProviderEndpointStep?: boolean
  allProviders: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  // Model defaults (set by app.tsx when advancing to models step)
  slotModels: Record<MagnitudeSlot, ModelSelection | null>
  connectedProviderName?: string | null
  selectedProviderId?: string | null
  selectedProviderDiscoveredModels?: Array<{ id: string; name?: string }>
  selectedProviderRememberedModelIds?: string[]
  totalSteps: number
  // Callbacks
  onProviderSelected: (providerId: string) => void
  onComplete: (result: Record<MagnitudeSlot, ModelSelection | null>) => void
  onBack: () => void
  onContinueFromLocalProvider?: () => void
  onLocalProviderSaveEndpoint?: (providerId: string, url: string) => void
  onLocalProviderRefreshModels?: (providerId: string) => void
  onLocalProviderAddManualModel?: (providerId: string, modelId: string) => void
  onLocalProviderRemoveManualModel?: (providerId: string, modelId: string) => void
  onLocalProviderSaveOptionalApiKey?: (providerId: string, apiKey: string) => void
  onSkip: () => void
  onWizardCtrlCExit: () => void
  providerSelectedIndex: number
  onProviderSelectedIndexChange: (index: number) => void
  onProviderHoverIndex?: (index: number) => void
  modelNavSelectedIndex: number
  onModelNavSelectedIndexChange: (index: number) => void
  onModelNavHoverIndex?: (index: number) => void
}

export const SetupWizardOverlay = memo(function SetupWizardOverlay({
  step,
  allProviders,
  detectedProviders,
  slotModels,
  connectedProviderName,
  selectedProviderId,
  selectedProviderDiscoveredModels = [],
  selectedProviderRememberedModelIds = [],
  totalSteps,
  onProviderSelected,
  onComplete,
  onBack,
  onContinueFromLocalProvider,
  onSkip,
  onWizardCtrlCExit,
  onLocalProviderSaveEndpoint,
  onLocalProviderRefreshModels,
  onLocalProviderAddManualModel,
  onLocalProviderRemoveManualModel,
  onLocalProviderSaveOptionalApiKey,
  providerSelectedIndex,
  onProviderSelectedIndexChange,
  onProviderHoverIndex,
  modelNavSelectedIndex,
  onModelNavSelectedIndexChange,
  onModelNavHoverIndex,
  hasProviderEndpointStep = false,
}: SetupWizardOverlayProps) {
  const theme = useTheme()
  const [backHovered, setBackHovered] = useState(false)
  const [confirmHovered, setConfirmHovered] = useState(false)

  const handleConfirm = useCallback(() => {
    if (!slotModels.lead) return
    onComplete(slotModels)
  }, [slotModels, onComplete])

  // Resolve model display names for all slots
  const slotDisplays = useMemo(() => {
    return Object.fromEntries(
      SLOT_UI_ORDER.map(({ slot }) => {
        const selection = slotModels[slot]
        if (!selection) return [slot, null]
        const provider = allProviders.find(p => p.id === selection.providerId)
        return [slot, { providerName: provider?.name ?? selection.providerId, modelId: selection.modelId }]
      })
    ) as Record<MagnitudeSlot, { providerName: string; modelId: string } | null>
  }, [slotModels, allProviders])

  const selectedProvider = selectedProviderId
    ? allProviders.find((provider) => provider.id === selectedProviderId)
    : null

  const discoveredProviderModels = selectedProviderDiscoveredModels
    .filter((m): m is { id: string; name?: string } => typeof m?.id === 'string' && m.id.trim().length > 0)
    .map((m) => ({ id: m.id, name: typeof m.name === 'string' && m.name.trim().length > 0 ? m.name : m.id }))

  const discoveredProviderModelIdSet = new Set(discoveredProviderModels.map((m) => m.id))
  const rememberedProviderModelIds = selectedProviderRememberedModelIds
    .filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
    .filter((id) => !discoveredProviderModelIdSet.has(id))

  const modelsSubtitle = connectedProviderName
    ? `${connectedProviderName} is connected! We have assigned default models for each role. You can edit these at anytime in /model.`
    : 'We have assigned default models for each role. You can edit these at anytime in /model.'

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option

    if (key.ctrl && key.name === 'c' && !key.meta && !key.option) {
      key.preventDefault()
      onWizardCtrlCExit()
      return
    }

    if (key.name === 'escape') {
      key.preventDefault()
      if (step !== 'provider') {
        onBack()
      }
      return
    }

    if (key.ctrl && key.name === 's' && !key.meta && !key.option && !key.shift) {
      key.preventDefault()
      onSkip()
      return
    }

    if (step === 'provider') {
      if (allProviders.length === 0) {
        key.preventDefault()
        return
      }
      if (key.name === 'up' && plain) {
        key.preventDefault()
        onProviderSelectedIndexChange(Math.max(0, providerSelectedIndex - 1))
        return
      }
      if (key.name === 'down' && plain) {
        key.preventDefault()
        onProviderSelectedIndexChange(Math.min(allProviders.length - 1, providerSelectedIndex + 1))
        return
      }
      if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        key.preventDefault()
        const provider = allProviders[providerSelectedIndex]
        if (provider) onProviderSelected(provider.id)
        return
      }
      key.preventDefault()
      return
    }

    if (step === 'local-provider') {
      if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        key.preventDefault()
        onContinueFromLocalProvider?.()
        return
      }
      return
    }

    if (step === 'models') {
      if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        key.preventDefault()
        handleConfirm()
        return
      }
      key.preventDefault()
      return
    }

    key.preventDefault()
  }, [
    step,
    onWizardCtrlCExit,
    onSkip,
    allProviders,
    providerSelectedIndex,
    onProviderSelectedIndexChange,
    onProviderSelected,
    onBack,
    modelNavSelectedIndex,
    onModelNavSelectedIndexChange,
    handleConfirm,
  ]))

  if (step === 'local-provider') {
    if (!selectedProvider || selectedProvider.providerFamily !== 'local') return null
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <WizardHeader
          stepLabel={`Local provider setup (2 of ${totalSteps})`}
          subtitle="Configure endpoint/models for your local provider before role assignments."
          onSkip={onSkip}
          theme={theme}
        />

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
          <LocalProviderPage
            providerName={selectedProvider.name}
            endpoint={selectedProvider.defaultBaseUrl ?? ''}
            endpointPlaceholder={selectedProvider.defaultBaseUrl ?? 'http://localhost:1234/v1'}
            discoveredModels={discoveredProviderModels}
            manualModelIds={rememberedProviderModelIds}
            showOptionalApiKey={selectedProvider.authMethods.some((method) => method.type === 'api-key')}
            showEndpointSaveButton={false}
            showApiKeySaveButton={false}
            onSaveEndpoint={(url) => onLocalProviderSaveEndpoint?.(selectedProvider.id, url)}
            onRefreshModels={() => onLocalProviderRefreshModels?.(selectedProvider.id)}
            onAddManualModel={(modelId) => onLocalProviderAddManualModel?.(selectedProvider.id, modelId)}
            onRemoveManualModel={(modelId) => onLocalProviderRemoveManualModel?.(selectedProvider.id, modelId)}
            onSaveOptionalApiKey={(apiKey) => onLocalProviderSaveOptionalApiKey?.(selectedProvider.id, apiKey)}
          />

          <box style={{ paddingTop: 1 }}>
            <Button onClick={() => onContinueFromLocalProvider?.()} onMouseOver={() => setConfirmHovered(true)} onMouseOut={() => setConfirmHovered(false)}>
              <box style={{
                borderStyle: 'single',
                borderColor: confirmHovered ? theme.success : theme.border,
                customBorderChars: BOX_CHARS,
                paddingLeft: 2,
                paddingRight: 2,
              }}>
                <text style={{ fg: theme.success }}>Continue to model assignment (Enter)</text>
              </box>
            </Button>
          </box>
        </scrollbox>

        <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={onBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
            <box style={{
              borderStyle: 'single',
              borderColor: backHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (Esc)</text>
            </box>
          </Button>
        </box>
      </box>
    )
  }

  if (step === 'models') {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <WizardHeader
          stepLabel={`Models (${hasProviderEndpointStep ? 3 : 2} of ${totalSteps})`}
          subtitle={modelsSubtitle}
          onSkip={onSkip}
          theme={theme}
        />

        {/* Model confirmation — scrollable for small terminals */}
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
          {SLOT_UI_ORDER.map(({ slot, label, description }) => {
            const display = slotDisplays[slot]
            return (
              <box key={slot} style={{ paddingBottom: 1 }}>
                <text style={{ fg: theme.foreground }}>
                  <span attributes={TextAttributes.BOLD}>{label}</span>{' '}
                  <span style={{ fg: theme.muted }}>{description}</span>
                  {display ? (
                    <>
                      <span style={{ fg: theme.muted }}>: </span>
                      <span style={{ fg: slate[300] }}>{display.providerName}</span>
                      <span style={{ fg: slate[300] }} attributes={TextAttributes.DIM}> · </span>
                      <span style={{ fg: slate[300] }}>{display.modelId}</span>
                    </>
                  ) : (
                    <span style={{ fg: theme.muted }}>: Not configured</span>
                  )}
                </text>
              </box>
            )
          })}

          {/* Start coding / Continue button */}
          <box style={{ paddingBottom: 1 }}>
            <Button onClick={handleConfirm} onMouseOver={() => setConfirmHovered(true)} onMouseOut={() => setConfirmHovered(false)}>
              <box style={{
                borderStyle: 'single',
                borderColor: confirmHovered ? theme.success : theme.border,
                customBorderChars: BOX_CHARS,
                paddingLeft: 2,
                paddingRight: 2,
              }}>
                <text style={{ fg: theme.success }}>
                  {modelNavSelectedIndex === 0 ? '> ' : '  '}{totalSteps > 2 ? 'Continue (Enter)' : 'Start coding (Enter)'}
                </text>
              </box>
            </Button>
          </box>
        </scrollbox>

        {/* Footer */}
        <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
          <Button onClick={onBack} onMouseOver={() => setBackHovered(true)} onMouseOut={() => setBackHovered(false)}>
            <box style={{
              borderStyle: 'single',
              borderColor: backHovered ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (Esc)</text>
            </box>
          </Button>
        </box>
      </box>
    )
  }

  // Step 1: Provider selection
  return (
    <box style={{ flexDirection: 'column', height: '100%' }}>
      <WizardHeader
        stepLabel={`Providers (1 of ${totalSteps})`}
        subtitle="Choose a provider to get started. You can always add more later in /settings."
        onSkip={onSkip}
        theme={theme}
      />

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
          },
        }}
      >
        {allProviders.map((provider, index) => {
          const detected = detectedProviders.find(d => d.provider.id === provider.id)
          const isConnected = !!detected
          const isSelected = index === providerSelectedIndex
          const description = PROVIDER_DESCRIPTIONS[provider.id]

          // Compute source label
          let detectedLabel = 'Connected'
          if (detected) {
            if (provider.providerFamily === 'local') {
              detectedLabel = provider.defaultBaseUrl ? 'Connected (Discovered)' : 'Connected (Configured)'
            } else if (detected.source === 'env') {
              detectedLabel = 'Connected (Env Var)'
            } else if (detected.source === 'stored') {
              if (detected.auth?.type === 'api') detectedLabel = 'Connected (API Key)'
              else if (detected.auth?.type === 'oauth') detectedLabel = 'Connected (Auth Token)'
              else if (detected.auth?.type === 'aws') detectedLabel = 'Connected (AWS Creds)'
              else if (detected.auth?.type === 'gcp') detectedLabel = 'Connected (GCP Creds)'
            }
          }

          return (
            <Button
              key={provider.id}
              onClick={() => onProviderSelected(provider.id)}
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
                  <span style={{ fg: theme.success }}>{' — '}{detectedLabel}</span>
                )}
              </text>
            </Button>
          )
        })}
      </scrollbox>

      {/* Footer */}
      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            ↑/↓ navigate  |  Enter select
          </span>
        </text>
      </box>
    </box>
  )
})
