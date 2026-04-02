import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ProviderDefinition } from '@magnitudedev/agent'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: '#fff',
    muted: '#888',
    primary: '#4af',
    success: '#6f6',
    border: '#444',
    surface: '#222',
    warning: '#ff0',
    error: '#f66',
    background: '#000',
  }),
}))

const { ProviderEndpointOverlay } = await import('./provider-endpoint-overlay')

function htmlToText(html: string): string {
  return html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
}

const noop = () => {}

test('provider endpoint wizard for local provider renders discovery/manual guidance and back affordance', () => {
  const provider = {
    id: 'lmstudio',
    name: 'LM Studio',
    providerFamily: 'local',
    defaultBaseUrl: 'http://localhost:1234/v1',
    authMethods: [{ type: 'none', label: 'Local endpoint' }],
    bamlProvider: 'openai-generic',
    models: [],
  } as ProviderDefinition

  const html = renderToStaticMarkup(
    <ProviderEndpointOverlay
      provider={provider}
      onSubmit={noop}
      onCancel={noop}
      wizardMode={{
        stepLabel: 'Provider setup (2 of 4)',
        subtitle: 'Configure endpoint',
        onSkip: noop,
        onBack: noop,
      }}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Endpoint is optional here. Save to refresh available models;')
  expect(text).toContain('Manual model ID (optional):')
  expect(text).toContain('← Back (B)')
})

test('provider endpoint for non-local provider keeps seed-model labeling', () => {
  const provider = {
    id: 'openai',
    name: 'OpenAI',
    providerFamily: 'cloud',
    authMethods: [{ type: 'none', label: 'Endpoint' }],
    bamlProvider: 'openai',
    models: [],
  } as ProviderDefinition

  const html = renderToStaticMarkup(
    <ProviderEndpointOverlay
      provider={provider}
      onSubmit={noop}
      onCancel={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Seed model (optional):')
  expect(text).not.toContain('discover models')
})
