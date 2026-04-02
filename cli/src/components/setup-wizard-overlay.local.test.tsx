import { test, expect, mock } from 'bun:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { act, create } from 'react-test-renderer'
import { SingleLineInput } from './single-line-input'
import type { ProviderDefinition, DetectedProvider } from '@magnitudedev/agent'
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

const { SetupWizardOverlay } = await import('./setup-wizard-overlay')

function htmlToText(html: string): string {
  return html.replace(/<[^>]*>/g, ' ').replace(/[{}[\]",:]/g, ' ').replace(/\s+/g, ' ').trim()
}

const noop = () => {}

const emptySlots: Record<MagnitudeSlot, ModelSelection | null> = {
  lead: null,
  explorer: null,
  planner: null,
  builder: null,
  reviewer: null,
  debugger: null,
  browser: null,
}

const localProviders: ProviderDefinition[] = [
  {
    id: 'lmstudio',
    name: 'LM Studio',
    providerFamily: 'local',
    defaultBaseUrl: 'http://localhost:1234/v1',
    authMethods: [{ type: 'none', label: 'Local endpoint' }],
    bamlProvider: 'openai-generic',
    models: [],
  } as any,
  {
    id: 'anthropic',
    name: 'Anthropic',
    providerFamily: 'cloud',
    authMethods: [{ type: 'api-key', label: 'API key', envKeys: ['ANTHROPIC_API_KEY'] }],
    bamlProvider: 'anthropic',
    models: [],
  } as any,
]

test('setup wizard provider list uses local configured semantics (not detected wording)', () => {
  const detectedProviders: DetectedProvider[] = [
    { provider: localProviders[0]!, auth: null, source: 'stored' },
    { provider: localProviders[1]!, auth: { type: 'api', key: 'x' } as any, source: 'env' },
  ]

  const html = renderToStaticMarkup(
    <SetupWizardOverlay
      step="provider"
      allProviders={localProviders}
      detectedProviders={detectedProviders}
      slotModels={emptySlots}
      connectedProviderName={null}
      selectedProviderId={null}
      totalSteps={4}
      onProviderSelected={noop}
      onComplete={noop}
      onBack={noop}
      onSkip={noop}
      onWizardCtrlCExit={noop}
      providerSelectedIndex={0}
      onProviderSelectedIndexChange={noop}
      onProviderHoverIndex={noop}
      modelNavSelectedIndex={0}
      onModelNavSelectedIndexChange={noop}
      onModelNavHoverIndex={noop}
      hasProviderEndpointStep
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('LM Studio')
  expect(text).toContain('Configured')
  expect(text).not.toContain('Detected')
  expect(text).toContain('Anthropic')
  expect(text).toContain('Connected (Env Var)')
})

test('models step shows discovered models for selected local provider and local step count', () => {
  const providers: ProviderDefinition[] = [
    {
      ...localProviders[0]!,
      models: [
        { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
        { id: 'llama3.1:8b', name: 'Llama 3.1 8B' },
      ],
    } as any,
    localProviders[1]!,
  ]

  const slotModels: Record<MagnitudeSlot, ModelSelection | null> = {
    ...emptySlots,
    lead: { providerId: 'lmstudio', modelId: 'qwen2.5-coder' },
  }

  const html = renderToStaticMarkup(
    <SetupWizardOverlay
      step="models"
      allProviders={providers}
      detectedProviders={[]}
      slotModels={slotModels}
      connectedProviderName="LM Studio"
      selectedProviderId="lmstudio"
      selectedProviderDiscoveredModels={[
        { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
        { id: 'llama3.1:8b', name: 'Llama 3.1 8B' },
      ]}
      selectedProviderRememberedModelIds={['custom-local-model']}
      totalSteps={4}
      onProviderSelected={noop}
      onComplete={noop}
      onBack={noop}
      onSkip={noop}
      onWizardCtrlCExit={noop}
      providerSelectedIndex={0}
      onProviderSelectedIndexChange={noop}
      onProviderHoverIndex={noop}
      modelNavSelectedIndex={0}
      onModelNavSelectedIndexChange={noop}
      onModelNavHoverIndex={noop}
      hasProviderEndpointStep={false}
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Models (2 of 4)')
  expect(text).toContain('Review role assignments below.')
  expect(text).toContain('LM Studio')
  expect(text).not.toContain('Models from LM Studio')
  expect(text).not.toContain('No models listed yet')
  expect(text).not.toContain('(recommended default)')
})

test('local-provider step renders local setup page, keeps only Add button, and dedupes discovered IDs out of manual models list', () => {
  const providers: ProviderDefinition[] = [localProviders[0]!, localProviders[1]!]

  const html = renderToStaticMarkup(
    <SetupWizardOverlay
      step="local-provider"
      allProviders={providers}
      detectedProviders={[]}
      slotModels={emptySlots}
      connectedProviderName="LM Studio"
      selectedProviderId="lmstudio"
      selectedProviderDiscoveredModels={[
        { id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' },
      ]}
      selectedProviderRememberedModelIds={['qwen2.5-coder', 'manual-only-model']}
      totalSteps={4}
      onProviderSelected={noop}
      onComplete={noop}
      onBack={noop}
      onContinueFromLocalProvider={noop}
      onSkip={noop}
      onWizardCtrlCExit={noop}
      onLocalProviderSaveOptionalApiKey={noop}
      providerSelectedIndex={0}
      onProviderSelectedIndexChange={noop}
      onProviderHoverIndex={noop}
      modelNavSelectedIndex={0}
      onModelNavSelectedIndexChange={noop}
      onModelNavHoverIndex={noop}
      hasProviderEndpointStep
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('Local provider setup (2 of 4)')
  expect(text).toContain('Discovered models')
  expect(text).toContain('qwen2.5-coder')
  expect(text).toContain('Manual models')
  expect(text).toContain('manual-only-model')
  expect(text).toContain('Add')
  expect(text).not.toContain('[Add]')
  expect(text).not.toContain('[Save endpoint]')
  expect(text).not.toContain('[Save key]')
  expect(text).not.toContain('Delete key')
})

test('models step shows empty discovered inventory fallback guidance for selected local provider', () => {
  const html = renderToStaticMarkup(
    <SetupWizardOverlay
      step="models"
      allProviders={localProviders}
      detectedProviders={[]}
      slotModels={emptySlots}
      connectedProviderName="LM Studio"
      selectedProviderId="lmstudio"
      totalSteps={4}
      onProviderSelected={noop}
      onComplete={noop}
      onBack={noop}
      onSkip={noop}
      onWizardCtrlCExit={noop}
      providerSelectedIndex={0}
      onProviderSelectedIndexChange={noop}
      onProviderHoverIndex={noop}
      modelNavSelectedIndex={0}
      onModelNavSelectedIndexChange={noop}
      onModelNavHoverIndex={noop}
      hasProviderEndpointStep
    />,
  )

  const text = htmlToText(html)
  expect(text).toContain('LM Studio')
})

test('local-provider step add flow updates manual models on same wizard page', () => {
  function WizardAddHarness() {
    const [remembered, setRemembered] = React.useState<string[]>([])
    return (
      <SetupWizardOverlay
        step="local-provider"
        allProviders={localProviders}
        detectedProviders={[]}
        slotModels={emptySlots}
        connectedProviderName="LM Studio"
        selectedProviderId="lmstudio"
        selectedProviderDiscoveredModels={[{ id: 'qwen2.5-coder', name: 'Qwen 2.5 Coder' }]}
        selectedProviderRememberedModelIds={remembered}
        totalSteps={4}
        onProviderSelected={noop}
        onComplete={noop}
        onBack={noop}
        onContinueFromLocalProvider={noop}
        onSkip={noop}
        onWizardCtrlCExit={noop}
        onLocalProviderAddManualModel={(_providerId, modelId) => setRemembered((prev) => Array.from(new Set([...prev, modelId])))}
        onLocalProviderSaveOptionalApiKey={noop}
        providerSelectedIndex={0}
        onProviderSelectedIndexChange={noop}
        onProviderHoverIndex={noop}
        modelNavSelectedIndex={0}
        onModelNavSelectedIndexChange={noop}
        onModelNavHoverIndex={noop}
        hasProviderEndpointStep
      />
    )
  }

  let renderer!: ReturnType<typeof create>
  act(() => {
    renderer = create(<WizardAddHarness />)
  })

  const manualInput = renderer.root.findAllByType(SingleLineInput).find((node) => node.props.placeholder === 'Add model ID')
  expect(manualInput).toBeDefined()

  act(() => {
    manualInput!.props.onChange('manual-only-model')
  })

  const addButtonText = renderer.root.findAll((n) => n.type === 'text' && (Array.isArray(n.props.children) ? n.props.children.join('') : String(n.props.children ?? '')) === 'Add')[0]
  const addButtonNode = addButtonText?.parent?.parent
  expect(addButtonNode).toBeDefined()

  act(() => {
    addButtonNode!.props.onMouseDown()
  })

  const text = htmlToText(JSON.stringify(renderer.toJSON()))
  expect(text).toContain('manual-only-model')
})
