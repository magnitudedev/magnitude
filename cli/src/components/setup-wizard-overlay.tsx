import { memo, useCallback, useMemo, useState } from 'react'
import { TextAttributes, type KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { WizardHeader } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import type { ProviderDefinition, ModelSelection } from '@magnitudedev/agent'
import type { DetectedProvider } from '@magnitudedev/agent'
import type { MagnitudeSlot } from '@magnitudedev/agent'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'github-copilot': '(GitHub.com or Enterprise)',
  local: '(Ollama, LM Studio, llama.cpp, vLLM)',
}

export const SLOT_UI_ORDER: { slot: MagnitudeSlot; label: string; description: string }[] = [
  { slot: 'lead', label: 'Team Lead', description: '(Coordinates all subagent usage)' },
  { slot: 'explorer', label: 'Explorer', description: '(Reads lots of files and does web searches)' },
  { slot: 'planner', label: 'Planner', description: '(Plans out implementation approaches)' },
  { slot: 'builder', label: 'Builder', description: '(Implements changes in files)' },
  { slot: 'reviewer', label: 'Reviewer', description: '(Reviews code for correctness)' },
  { slot: 'debugger', label: 'Debugger', description: '(Root causes and fixes issues)' },
  { slot: 'browser', label: 'Browser', description: '(Visually navigates a browser)' },
]

export type WizardStep = 'provider' | 'models' | 'browser'

interface SetupWizardOverlayProps {
  step: WizardStep
  allProviders: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  // Model defaults (set by app.tsx when advancing to models step)
  slotModels: Record<MagnitudeSlot, ModelSelection | null>
  connectedProviderName?: string | null
  totalSteps: number
  // Callbacks
  onProviderSelected: (providerId: string) => void
  onComplete: (result: Record<MagnitudeSlot, ModelSelection | null>) => void
  onBack: () => void
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
  totalSteps,
  onProviderSelected,
  onComplete,
  onBack,
  onSkip,
  onWizardCtrlCExit,
  providerSelectedIndex,
  onProviderSelectedIndexChange,
  onProviderHoverIndex,
  modelNavSelectedIndex,
  onModelNavSelectedIndexChange,
  onModelNavHoverIndex,
}: SetupWizardOverlayProps) {
  const theme = useTheme()
  const [backHovered, setBackHovered] = useState(false)

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

  const modelsSubtitle = connectedProviderName
    ? `You've successfully connected ${connectedProviderName}! We've configured default models based on your provider. You can change these anytime with /models.`
    : 'Your default models have been configured. You can change these anytime with /models.'

  useKeyboard(useCallback((key: KeyEvent) => {
    const plain = !key.ctrl && !key.meta && !key.option

    if (key.ctrl && key.name === 'c' && !key.meta && !key.option) {
      key.preventDefault()
      onWizardCtrlCExit()
      return
    }

    if (key.name === 'escape') {
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

    if (step === 'models') {
      if (key.name === 'b' && plain && !key.shift) {
        key.preventDefault()
        onBack()
        return
      }
      if (key.name === 'up' && plain) {
        key.preventDefault()
        onModelNavSelectedIndexChange(Math.max(0, modelNavSelectedIndex - 1))
        return
      }
      if (key.name === 'down' && plain) {
        key.preventDefault()
        onModelNavSelectedIndexChange(Math.min(7, modelNavSelectedIndex + 1))
        return
      }
      if ((key.name === 'return' || key.name === 'enter') && plain && !key.shift) {
        key.preventDefault()
        if (modelNavSelectedIndex === 0) handleConfirm()
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

  if (step === 'models') {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <WizardHeader
          stepLabel={`Models (2 of ${totalSteps})`}
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
          {/* Start chatting / Continue button */}
          <box style={{ paddingBottom: 1 }}>
            <Button onClick={handleConfirm}>
              <box style={{
                borderStyle: 'single',
                borderColor: modelNavSelectedIndex === 0 ? theme.success : theme.border,
                customBorderChars: BOX_CHARS,
                paddingLeft: 2,
                paddingRight: 2,
              }}>
                <text style={{ fg: modelNavSelectedIndex === 0 ? theme.success : theme.foreground }}>
                  {modelNavSelectedIndex === 0 ? '> ' : '  '}{totalSteps > 2 ? 'Continue (Enter)' : 'Start chatting (Enter)'}
                </text>
              </box>
            </Button>
          </box>

          {SLOT_UI_ORDER.map(({ slot, label, description }, idx) => {
            const display = slotDisplays[slot]
            const navIndex = idx + 1 // 0 is confirm button, 1-7 are slots
            return (
              <box key={slot} style={{ flexDirection: 'column', paddingBottom: 1 }}>
                <box style={{ paddingBottom: 0 }}>
                  <text style={{ fg: theme.foreground }}>
                    <span attributes={TextAttributes.BOLD}>{label}</span> {description}
                  </text>
                </box>
                <box style={{
                  flexDirection: 'row',
                  borderStyle: 'single',
                  borderColor: modelNavSelectedIndex === navIndex ? theme.primary : theme.border,
                  customBorderChars: BOX_CHARS,
                  paddingLeft: 1,
                  paddingRight: 1,
                }}>
                  <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                    {display ? (
                      <>
                        {display.providerName}
                        <span attributes={TextAttributes.DIM}> · </span>
                        {display.modelId}
                      </>
                    ) : (
                      <span style={{ fg: theme.muted }}>Not configured</span>
                    )}
                  </text>
                </box>
              </box>
            )
          })}
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
              <text style={{ fg: backHovered ? theme.primary : theme.muted }}>← Back (B)</text>
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

          // Compute detection source label
          let detectedLabel = 'Detected'
          if (detected) {
            if (detected.source === 'env') {
              detectedLabel = 'Detected (Env Var)'
            } else if (detected.source === 'stored') {
              if (detected.auth?.type === 'api') detectedLabel = 'Detected (API Key)'
              else if (detected.auth?.type === 'oauth') detectedLabel = 'Detected (Auth Token)'
              else if (detected.auth?.type === 'aws') detectedLabel = 'Detected (AWS Creds)'
              else if (detected.auth?.type === 'gcp') detectedLabel = 'Detected (GCP Creds)'
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
