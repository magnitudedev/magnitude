import { test, expect, mock } from 'bun:test'
import React from 'react'
import TestRenderer, { act } from 'react-test-renderer'
import type { ProviderDefinition } from '@magnitudedev/agent'
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

function treeText(node: any): string {
  const out: string[] = []
  const walk = (n: any) => {
    if (!n) return
    if (typeof n === 'string') out.push(n)
    if (Array.isArray(n)) {
      for (const c of n) walk(c)
      return
    }
    walk(n.children)
  }
  walk(node)
  return out.join(' ').replace(/\s+/g, ' ').trim()
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

test('load preset modal includes local provider defaults rows', () => {
  const lmstudio: ProviderDefinition = {
    id: 'lmstudio',
    name: 'LM Studio',
    providerFamily: 'local',
    defaultBaseUrl: 'http://localhost:1234/v1',
    authMethods: [{ type: 'none', label: 'No auth required' }],
    bamlProvider: 'openai-generic',
    models: [],
  } as any

  const ollama: ProviderDefinition = {
    id: 'ollama',
    name: 'Ollama',
    providerFamily: 'local',
    defaultBaseUrl: 'http://localhost:11434/v1',
    authMethods: [{ type: 'none', label: 'No auth required' }],
    bamlProvider: 'openai-generic',
    models: [],
  } as any

  let renderer!: TestRenderer.ReactTestRenderer
  act(() => {
    renderer = TestRenderer.create(
      <SettingsOverlay
        activeTab="model"
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
        allProviders={[lmstudio, ollama]}
        detectedProviders={[
          { provider: lmstudio, auth: null, source: 'stored' } as any,
          { provider: ollama, auth: null, source: 'stored' } as any,
        ]}
        providerSelectedIndex={0}
        onProviderSelect={noop}
        onProviderHoverIndex={noop}
        providerDetailStatus={null}
        providerDetailOptions={undefined}
        providerDetailActions={[]}
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
  })

  const buttons = renderer.root.findAll((n) => typeof n.props?.onClick === 'function')
  const loadPresetButton = buttons.find((n) => treeText(n).includes('Load preset'))
  expect(loadPresetButton).toBeDefined()

  act(() => {
    loadPresetButton!.props.onClick()
  })

  const output = treeText(renderer.toJSON())
  expect(output).toContain('LM Studio defaults')
  expect(output).toContain('Ollama defaults')
})

test('load preset modal keeps non-local provider defaults behavior', () => {
  const anthropic: ProviderDefinition = {
    id: 'anthropic',
    name: 'Anthropic',
    providerFamily: 'cloud',
    authMethods: [{ type: 'api-key', label: 'API key', envKeys: ['ANTHROPIC_API_KEY'] }],
    bamlProvider: 'anthropic',
    models: [],
  } as any

  const onLoadCalls: Array<[string, string | undefined]> = []
  const onLoadPreset = async (name: string, preferredProviderId?: string) => {
    onLoadCalls.push([name, preferredProviderId])
  }

  let renderer!: TestRenderer.ReactTestRenderer
  act(() => {
    renderer = TestRenderer.create(
      <SettingsOverlay
        activeTab="model"
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
        allProviders={[anthropic]}
        detectedProviders={[{ provider: anthropic, auth: { type: 'api', key: 'x' }, source: 'stored' } as any]}
        providerSelectedIndex={0}
        onProviderSelect={noop}
        onProviderHoverIndex={noop}
        providerDetailStatus={null}
        providerDetailOptions={undefined}
        providerDetailActions={[]}
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
        onLoadPreset={onLoadPreset}
        onDeletePreset={noopAsync}
      />,
    )
  })

  const buttons = renderer.root.findAll((n) => typeof n.props?.onClick === 'function')
  const loadPresetButton = buttons.find((n) => treeText(n).includes('Load preset'))
  expect(loadPresetButton).toBeDefined()

  act(() => {
    loadPresetButton!.props.onClick()
  })

  const providerRow = renderer.root
    .findAll((n) => typeof n.props?.onClick === 'function')
    .find((n) => treeText(n).includes('Anthropic defaults'))
  expect(providerRow).toBeDefined()

  act(() => {
    providerRow!.props.onClick()
  })

  expect(onLoadCalls).toEqual([['__system_defaults__', 'anthropic']])
})
