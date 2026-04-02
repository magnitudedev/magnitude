import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

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
    info: '#0ff',
  }),
}))

const { LocalProviderPage } = await import('./local-provider-page')

const noop = () => {}

function htmlToText(html: string): string {
  return html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
}

test('provider variant shows all three action buttons', () => {
  const html = renderToStaticMarkup(
    <LocalProviderPage
      providerName="LM Studio"
      endpoint="http://localhost:1234/v1"
      endpointPlaceholder="http://localhost:1234/v1"
      discoveredModels={[]}
      manualModelIds={[]}
      showOptionalApiKey
      optionalApiKeyValue=""
      onSaveEndpoint={noop}
      onRefreshModels={noop}
      onAddManualModel={noop}
      onRemoveManualModel={noop}
      onSaveOptionalApiKey={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Save endpoint')
  expect(text).toContain('Add')
  expect(text).toContain('Save key')
  expect(text).not.toContain('[Save endpoint]')
  expect(text).not.toContain('[Add]')
  expect(text).not.toContain('[Save key]')
})

test('wizard variant keeps only Add button', () => {
  const html = renderToStaticMarkup(
    <LocalProviderPage
      providerName="LM Studio"
      endpoint="http://localhost:1234/v1"
      endpointPlaceholder="http://localhost:1234/v1"
      discoveredModels={[]}
      manualModelIds={[]}
      showOptionalApiKey
      optionalApiKeyValue=""
      showEndpointSaveButton={false}
      showApiKeySaveButton={false}
      onSaveEndpoint={noop}
      onRefreshModels={noop}
      onAddManualModel={noop}
      onRemoveManualModel={noop}
      onSaveOptionalApiKey={noop}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Add')
  expect(text).not.toContain('[Add]')
  expect(text).not.toContain('Save endpoint')
  expect(text).not.toContain('Save key')
})
