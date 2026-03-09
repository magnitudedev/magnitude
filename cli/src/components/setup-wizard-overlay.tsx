import { memo, useCallback, useMemo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { WizardHeader } from './wizard-header'
import { BOX_CHARS } from '../utils/ui-constants'
import type { ProviderDefinition, ModelSelection } from '@magnitudedev/agent'
import type { DetectedProvider } from '@magnitudedev/agent'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'github-copilot': '(GitHub.com or Enterprise)',
  local: '(Ollama, LM Studio, llama.cpp, vLLM)',
}

export type WizardStep = 'provider' | 'models' | 'browser'

interface SetupWizardOverlayProps {
  step: WizardStep
  allProviders: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  // Model defaults (set by app.tsx when advancing to models step)
  primaryModel: ModelSelection | null
  secondaryModel: ModelSelection | null
  browserModel: ModelSelection | null
  connectedProviderName?: string | null
  totalSteps: number
  // Callbacks
  onProviderSelected: (providerId: string) => void
  onComplete: (result: { primaryModel: ModelSelection; secondaryModel: ModelSelection; browserModel: ModelSelection | null }) => void
  onBack: () => void
  onSkip: () => void
  // Navigation state (managed by app.tsx keyboard handler)
  providerSelectedIndex: number
  onProviderHoverIndex?: (index: number) => void
  modelNavSelectedIndex: number
  onModelNavHoverIndex?: (index: number) => void
}

export const SetupWizardOverlay = memo(function SetupWizardOverlay({
  step,
  allProviders,
  detectedProviders,
  primaryModel,
  secondaryModel,
  browserModel,
  connectedProviderName,
  totalSteps,
  onProviderSelected,
  onComplete,
  onBack,
  onSkip,
  providerSelectedIndex,
  onProviderHoverIndex,
  modelNavSelectedIndex,
  onModelNavHoverIndex,
}: SetupWizardOverlayProps) {
  const theme = useTheme()
  const [backHovered, setBackHovered] = useState(false)

  const handleConfirm = useCallback(() => {
    if (!primaryModel || !secondaryModel) return
    onComplete({ primaryModel, secondaryModel, browserModel: browserModel ?? null })
  }, [primaryModel, secondaryModel, browserModel, onComplete])

  // Resolve model display names
  const primaryDisplay = useMemo(() => {
    if (!primaryModel) return null
    const provider = allProviders.find(p => p.id === primaryModel.providerId)
    return { providerName: provider?.name ?? primaryModel.providerId, modelId: primaryModel.modelId }
  }, [primaryModel, allProviders])

  const secondaryDisplay = useMemo(() => {
    if (!secondaryModel) return null
    const provider = allProviders.find(p => p.id === secondaryModel.providerId)
    return { providerName: provider?.name ?? secondaryModel.providerId, modelId: secondaryModel.modelId }
  }, [secondaryModel, allProviders])

  const browserDisplay = useMemo(() => {
    if (!browserModel) return null
    const provider = allProviders.find(p => p.id === browserModel.providerId)
    return { providerName: provider?.name ?? browserModel.providerId, modelId: browserModel.modelId }
  }, [browserModel, allProviders])

  const modelsSubtitle = connectedProviderName
    ? `You've successfully connected ${connectedProviderName}! We've configured default models based on your provider. You can change these anytime with /models.`
    : 'Your default models have been configured. You can change these anytime with /models.'

  if (step === 'models') {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <WizardHeader
          stepLabel={`Models (2 of ${totalSteps})`}
          subtitle={modelsSubtitle}
          onSkip={onSkip}
          theme={theme}
        />

        {/* Model confirmation */}
        <box style={{ flexDirection: 'column', paddingLeft: 2, paddingRight: 2, paddingTop: 1, flexGrow: 1 }}>

          {/* Primary Model */}
          <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
            <box style={{ paddingBottom: 0 }}>
              <text style={{ fg: theme.foreground }}>
                <span attributes={TextAttributes.BOLD}>Primary Model</span>
                <span attributes={TextAttributes.DIM}> (Smarter)</span>
              </text>
            </box>
            <box style={{
              flexDirection: 'row',
              borderStyle: 'single',
              borderColor: modelNavSelectedIndex === 1 ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                {primaryDisplay ? (
                  <>
                    {primaryDisplay.providerName}
                    <span attributes={TextAttributes.DIM}> · </span>
                    {primaryDisplay.modelId}
                  </>
                ) : (
                  <span style={{ fg: theme.muted }}>Not configured</span>
                )}
              </text>
            </box>
          </box>

          {/* Secondary Model */}
          <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
            <box style={{ paddingBottom: 0 }}>
              <text style={{ fg: theme.foreground }}>
                <span attributes={TextAttributes.BOLD}>Secondary Model</span>
                <span attributes={TextAttributes.DIM}> (Faster)</span>
              </text>
            </box>
            <box style={{
              flexDirection: 'row',
              borderStyle: 'single',
              borderColor: modelNavSelectedIndex === 2 ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                {secondaryDisplay ? (
                  <>
                    {secondaryDisplay.providerName}
                    <span attributes={TextAttributes.DIM}> · </span>
                    {secondaryDisplay.modelId}
                  </>
                ) : (
                  <span style={{ fg: theme.muted }}>Not configured</span>
                )}
              </text>
            </box>
          </box>

          {/* Browser Model */}
          <box style={{ flexDirection: 'column', paddingBottom: 1 }}>
            <box style={{ paddingBottom: 0 }}>
              <text style={{ fg: theme.foreground }}>
                <span attributes={TextAttributes.BOLD}>Browser Agent Model</span>
                <span attributes={TextAttributes.DIM}> (Vision)</span>
              </text>
            </box>
            <box style={{
              flexDirection: 'row',
              borderStyle: 'single',
              borderColor: modelNavSelectedIndex === 3 ? theme.primary : theme.border,
              customBorderChars: BOX_CHARS,
              paddingLeft: 1,
              paddingRight: 1,
            }}>
              <text style={{ fg: theme.foreground, flexGrow: 1 }}>
                {browserDisplay ? (
                  <>
                    {browserDisplay.providerName}
                    <span attributes={TextAttributes.DIM}> · </span>
                    {browserDisplay.modelId}
                  </>
                ) : (
                  <span style={{ fg: theme.muted }}>None detected</span>
                )}
              </text>
            </box>
          </box>

          {/* Start chatting / Continue button */}
          <box style={{ paddingTop: 1 }}>
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
        </box>

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
