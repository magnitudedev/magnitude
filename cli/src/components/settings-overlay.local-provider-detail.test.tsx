import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { ProviderDefinition, ProviderAuthMethodStatus } from '@magnitudedev/agent'
import type { MagnitudeSlot, ModelSelection } from '@magnitudedev/agent'

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

const { SettingsOverlay } = await import('./settings-overlay')

function htmlToText(html: string): string {
  return html
    .replace(/<[^>]*>/g, ' ')
    .replace(/&quot;/g, '"')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&')
    .replace(/&#x27;/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
}

const noop = () => {}
const noopAsync = async () => {}

const emptySlots: Record<MagnitudeSlot, ModelSelection | null> = {
  lead: null,
  explorer: null,
  planner: null,
  builder: null,
  reviewer: null,
  debugger: null,
  browser: null,
}

function buildLocalProviderStatus(models: Array<{ id: string; name: string }>): ProviderAuthMethodStatus {
  const provider: ProviderDefinition = {
    id: 'lmstudio',
    name: 'LM Studio',
    providerFamily: 'local',
    defaultBaseUrl: 'http://localhost:1234/v1',
    authMethods: [{ type: 'none', label: 'Local endpoint' }],
    bamlProvider: 'openai-generic',
    models: models as any,
  } as ProviderDefinition

  return {
    provider,
    methods: [
      {
        methodIndex: 0,
        method: { type: 'none', label: 'Local endpoint' },
        connected: true,
        auth: null,
        source: 'stored',
      },
    ],
  } as ProviderAuthMethodStatus
}

function renderDetail(params: {
  status: ProviderAuthMethodStatus
  options?: Record<string, unknown>
  actions?: Array<{ type: string; methodIndex: number; label: string }>
}): string {
  const html = renderToStaticMarkup(
    <SettingsOverlay
      activeTab="provider"
      onTabChange={noop}
      onClose={noop}
      modelItems={[]}
      modelSelectedIndex={0}
      onModelSelect={noop}
      onModelHoverIndex={noop}
      modelSearch=""
      onModelSearchChange={noop}
      showAllProviders={false}
      onToggleShowAllProviders={noop}
      showRecommendedOnly={false}
      onToggleShowRecommendedOnly={noop}
      allProviders={[params.status.provider]}
      detectedProviders={[]}
      providerSelectedIndex={0}
      onProviderSelect={noop}
      onProviderHoverIndex={noop}
      providerDetailStatus={params.status}
      providerDetailOptions={params.options as any}
      providerDetailActions={params.actions ?? []}
      providerDetailSelectedIndex={0}
      onProviderDetailAction={noop}
      onProviderDetailHoverIndex={noop}
      onLocalProviderSaveEndpoint={noop}
      onLocalProviderRefreshModels={noop}
      onLocalProviderAddManualModel={noop}
      onLocalProviderRemoveManualModel={noop}
      onLocalProviderSaveOptionalApiKey={noop}
      slotModels={emptySlots}
      selectingModelFor={null}
      onChangeSlot={noop}
      modelPrefsSelectedIndex={0}
      onModelPrefsHoverIndex={noop}
      onModelHandleKeyEvent={() => false}
      onProviderHandleKeyEvent={() => false}
      onBackFromModelPicker={noop}
      onBackFromProviderDetail={noop}
      presets={[]}
      systemDefaultsPresetToken="__system_defaults__"
      onSavePreset={noopAsync}
      onLoadPreset={noopAsync}
      onDeletePreset={noopAsync}
    />,
  )
  return htmlToText(html)
}

test('provider detail renders local shared page with endpoint/refresh and provider-only save buttons', () => {
  const text = renderDetail({
    status: buildLocalProviderStatus([]),
    options: { baseUrl: '   ' },
  })

  expect(text).toContain('Endpoint')
  expect(text).toContain('http://localhost:1234/v1')
  expect(text).toContain('No discovered models yet')
  expect(text).toContain('Save endpoint')
  expect(text).toContain('[Refresh]')

  const lmStudioMatches = text.match(/LM Studio/g) ?? []
  expect(lmStudioMatches.length).toBe(1)
})

test('provider detail renders configured endpoint, discovery status, discovered models, remembered/manual models', () => {
  const text = renderDetail({
    status: buildLocalProviderStatus([
      { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
      { id: 'llama3.1:8b', name: 'Llama 3.1 8B' },
    ]),
    options: {
      baseUrl: 'http://localhost:1234/v1',
      discoveredModels: [
        { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
        { id: 'llama3.1:8b', name: 'Llama 3.1 8B' },
      ],
      rememberedModelIds: ['custom-local-model'],
    },
  })

  expect(text).toContain('Endpoint')
  expect(text).toContain('http://localhost:1234/v1')
  expect(text).toContain('Discovered 2 models')
  expect(text).toContain('Discovered models')
  expect(text).toContain('Qwen 2.5 Coder')
  expect(text).toContain('qwen2.5-coder')
  expect(text).toContain('Llama 3.1 8B')
  expect(text).toContain('llama3.1:8b')
  expect(text).toContain('Manual models')
  expect(text).toContain('custom-local-model')
})

test('provider detail renders configured endpoint with no discovered models state', () => {
  const text = renderDetail({
    status: buildLocalProviderStatus([]),
    options: { baseUrl: 'http://localhost:1234/v1', discoveredModels: [] },
  })

  expect(text).toContain('Endpoint')
  expect(text).toContain('http://localhost:1234/v1')
  expect(text).toContain('No discovered models yet')
})

test('provider detail distinguishes discovery failure from zero-model success', () => {
  const text = renderDetail({
    status: buildLocalProviderStatus([]),
    options: {
      baseUrl: 'http://localhost:1234/v1',
      discoveredModels: [],
      lastDiscoveryError: 'HTTP 500 for http://localhost:1234/v1/models',
    },
  })

  expect(text).toContain("Couldn't refresh models right now")
  expect(text).toContain('Note: HTTP 500 for http://localhost:1234/v1/models')
})

test('local provider detail surfaces optional API-key input with save label when no key is saved', () => {
  const text = renderDetail({
    status: {
      ...buildLocalProviderStatus([]),
      methods: [
        {
          methodIndex: 0,
          method: { type: 'none', label: 'Local endpoint' } as any,
          connected: true,
          auth: null,
          source: 'stored',
        },
        {
          methodIndex: 1,
          method: { type: 'api-key', label: 'Optional API key', envKeys: ['LMSTUDIO_API_KEY'] } as any,
          connected: false,
          auth: null,
          source: 'none',
        },
      ],
    } as any,
  })

  expect(text).toContain('API Key (Optional)')
  expect(text).toContain('Save key')
  expect(text).not.toContain('Delete key')
  expect(text).not.toContain('Disconnect')
  expect(text).not.toContain('Update')
  expect(text).not.toContain('Key:')
})

test('local provider detail shows delete label when optional API key is already saved', () => {
  const text = renderDetail({
    status: {
      ...buildLocalProviderStatus([]),
      methods: [
        {
          methodIndex: 0,
          method: { type: 'none', label: 'Local endpoint' } as any,
          connected: true,
          auth: null,
          source: 'stored',
        },
        {
          methodIndex: 1,
          method: { type: 'api-key', label: 'Optional API key', envKeys: ['LMSTUDIO_API_KEY'] } as any,
          connected: true,
          auth: { type: 'api', key: 'saved-key' } as any,
          source: 'stored',
        },
      ],
    } as any,
  })

  expect(text).toContain('API Key (Optional)')
  expect(text).toContain('Delete key')
  expect(text).not.toContain('Save key')
  expect(text).not.toContain('Disconnect')
  expect(text).not.toContain('Update')
  expect(text).not.toContain('Key:')
})

test('local provider detail excludes discovered model IDs from manual models section', () => {
  const text = renderDetail({
    status: buildLocalProviderStatus([
      { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
    ]),
    options: {
      baseUrl: 'http://localhost:1234/v1',
      discoveredModels: [{ id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' }],
      rememberedModelIds: ['qwen2.5-coder', 'manual-only-model'],
    },
  })

  expect(text).toContain('Discovered models')
  expect(text).toContain('qwen2.5-coder')
  expect(text).toContain('Manual models')
  expect(text).toContain('manual-only-model')
})
