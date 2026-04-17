import { memo } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import type { ProviderDefinition, DetectedProvider } from '@magnitudedev/agent'

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  anthropic: '(Claude Max or API key)',
  openai: '(ChatGPT Plus/Pro or API key)',
  'openai-compatible-local': '(DIY OpenAI-compatible local)',
}

interface ProviderSelectOverlayProps {
  providers: ProviderDefinition[]
  detectedProviders: DetectedProvider[]
  selectedIndex: number
  onSelect: (providerId: string) => void
  onHoverIndex?: (index: number) => void
  onClose: () => void
}

export const ProviderSelectOverlay = memo(function ProviderSelectOverlay({
  providers,
  detectedProviders,
  selectedIndex,
  onSelect,
  onHoverIndex,
  onClose,
}: ProviderSelectOverlayProps) {
  const theme = useTheme()

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
          <span attributes={TextAttributes.BOLD}>Providers</span>
        </text>
        <box style={{ flexDirection: 'row' }}>
          <Button onClick={onClose}>
            <text style={{ fg: theme.muted }} attributes={TextAttributes.UNDERLINE}>Close</text>
          </Button>
          <text style={{ fg: theme.muted }}>
            <span attributes={TextAttributes.DIM}>{' '}(Esc)  |  Enter to connect</span>
          </text>
        </box>
      </box>

      {/* Divider */}
      <box style={{ paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.border }}>
          {'─'.repeat(80)}
        </text>
      </box>

      {/* Scrollable provider list */}
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
        {providers.map((provider, index) => {
          const detected = detectedProviders.find(d => d.provider.id === provider.id)
          const isConnected = !!detected
          const isSelected = index === selectedIndex
          const description = PROVIDER_DESCRIPTIONS[provider.id]

          const sourceLabel = detected
            ? provider.providerFamily === 'local'
              ? (provider.defaultBaseUrl ? 'Discovered' : 'Configured')
              : detected.auth?.type === 'oauth' ? 'Subscription'
              : detected.auth?.type === 'api' ? 'API Key'
              : detected.auth?.type === 'aws' ? 'AWS Credentials'
              : detected.auth?.type === 'gcp' ? 'GCP Credentials'
              : 'No Auth Needed'
            : null

          return (
            <Button
              key={provider.id}
              onClick={() => onSelect(provider.id)}
              onMouseOver={() => onHoverIndex?.(index)}
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
                {provider.id === 'magnitude' && (
                  <span style={{ fg: theme.primary }}>{' '}(recommended)</span>
                )}
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

      {/* Footer */}
      <box style={{ paddingLeft: 2, paddingTop: 1, paddingBottom: 1, flexShrink: 0 }}>
        <text style={{ fg: theme.muted }}>
          <span attributes={TextAttributes.DIM}>
            {providers.length} provider{providers.length === 1 ? '' : 's'}
            {' · '}
            {detectedProviders.length} connected
          </span>
        </text>
      </box>
    </box>
  )
})
