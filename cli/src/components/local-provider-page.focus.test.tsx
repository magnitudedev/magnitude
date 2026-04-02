import { test, expect, mock } from 'bun:test'
import { act, create } from 'react-test-renderer'
import type { KeyEvent } from '@opentui/core'

let keyboardHandler: ((key: KeyEvent) => void) | null = null

const MockSingleLineInput = ({ value, placeholder, focused, onChange }: { value: string; placeholder?: string; focused?: boolean; onChange: (value: string) => void }) => (
  <box>
    <text>{`input:${placeholder ?? ''}:${focused ? 'focused' : 'unfocused'}:${value}`}</text>
    <box onMouseDown={() => onChange(value)} />
  </box>
)

mock.module('@opentui/react', () => ({
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboardHandler = handler
  },
}))

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

mock.module('./single-line-input', () => ({
  SingleLineInput: MockSingleLineInput,
}))

const { LocalProviderPage } = await import('./local-provider-page')

function renderPage(overrides?: Partial<React.ComponentProps<typeof LocalProviderPage>>) {
  const onSaveEndpoint = overrides?.onSaveEndpoint ?? (() => {})
  const onAddManualModel = overrides?.onAddManualModel ?? (() => {})
  const onRefreshModels = overrides?.onRefreshModels ?? (() => {})
  const onRemoveManualModel = overrides?.onRemoveManualModel ?? (() => {})

  let renderer!: ReturnType<typeof create>
  act(() => {
    renderer = create(
      <LocalProviderPage
        providerName="LM Studio"
        endpoint="http://localhost:1234/v1"
        endpointPlaceholder="http://localhost:1234/v1"
        discoveredModels={[]}
        manualModelIds={[]}
        onSaveEndpoint={onSaveEndpoint}
        onRefreshModels={onRefreshModels}
        onAddManualModel={onAddManualModel}
        onRemoveManualModel={onRemoveManualModel}
        {...overrides}
      />,
    )
  })
  return renderer
}

function textContent(renderer: ReturnType<typeof create>): string {
  const texts = renderer.root.findAll((n) => n.type === 'text')
  return texts.map((n) => {
    const child = n.props.children
    return Array.isArray(child) ? child.join('') : String(child ?? '')
  }).join(' | ')
}

test('neither endpoint nor manual input is focused by default', () => {
  const renderer = renderPage()
  const output = textContent(renderer)
  expect(output).toContain('input:http://localhost:1234/v1:unfocused:http://localhost:1234/v1')
  expect(output).toContain('input:Add model ID:unfocused:')
})

test('only one input focus is active at a time when switching fields', () => {
  const renderer = renderPage()

  const focusBoxes = renderer.root.findAll(
    (n) => n.type === 'box' && typeof n.props.onMouseDown === 'function' && n.props.style?.borderStyle === 'single',
  )
  expect(focusBoxes.length).toBeGreaterThanOrEqual(2)

  act(() => {
    focusBoxes[0]!.props.onMouseDown()
  })
  let output = textContent(renderer)
  expect(output).toContain('input:http://localhost:1234/v1:focused:http://localhost:1234/v1')
  expect(output).toContain('input:Add model ID:unfocused:')

  act(() => {
    focusBoxes[1]!.props.onMouseDown()
  })
  output = textContent(renderer)
  expect(output).toContain('input:http://localhost:1234/v1:unfocused:http://localhost:1234/v1')
  expect(output).toContain('input:Add model ID:focused:')
})

test('enter submits active field action: endpoint saves, manual adds (independent of button visibility)', () => {
  const saveCalls: string[] = []
  const addCalls: string[] = []

  const renderer = renderPage({
    onSaveEndpoint: (value) => saveCalls.push(value),
    onAddManualModel: (value) => addCalls.push(value),
  })

  const focusBoxes = renderer.root.findAll(
    (n) => n.type === 'box' && typeof n.props.onMouseDown === 'function' && n.props.style?.borderStyle === 'single',
  )
  const inputs = renderer.root.findAllByType(MockSingleLineInput)
  expect(inputs.length).toBe(2)

  act(() => {
    inputs[0]!.props.onChange('http://localhost:2222/v1')
    focusBoxes[0]!.props.onMouseDown()
  })

  act(() => {
    keyboardHandler?.({ name: 'enter', preventDefault: () => {} } as KeyEvent)
  })
  expect(saveCalls).toEqual(['http://localhost:2222/v1'])

  act(() => {
    inputs[1]!.props.onChange('qwen2.5-coder')
    focusBoxes[1]!.props.onMouseDown()
  })

  act(() => {
    keyboardHandler?.({ name: 'enter', preventDefault: () => {} } as KeyEvent)
  })
  expect(addCalls).toEqual(['qwen2.5-coder'])
})

test('refresh and remove render as inline bracket actions (not bordered button boxes)', () => {
  const renderer = renderPage({
    manualModelIds: ['manual-model-1'],
  })
  const text = textContent(renderer)
  expect(text).toContain('[Refresh]')
  expect(text).toContain('[Remove]')

  const borderedBoxes = renderer.root.findAll(
    (n) => n.type === 'box' && n.props.style?.borderStyle === 'single',
  )
  expect(borderedBoxes.length).toBeGreaterThanOrEqual(4)
  expect(borderedBoxes.length).toBeLessThan(7)
})
