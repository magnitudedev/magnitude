import { test, expect, mock } from 'bun:test'
import { act, create } from 'react-test-renderer'

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
  SingleLineInput: ({ value }: { value: string }) => <text>{value}</text>,
}))

mock.module('./button', () => ({
  Button: ({ children, onClick, onMouseOver, onMouseOut }: any) => (
    <box onMouseDown={onClick} onMouseOver={onMouseOver} onMouseOut={onMouseOut}>
      {children}
    </box>
  ),
}))

const { LocalProviderPage } = await import('./local-provider-page')

function renderPage() {
  return create(
    <LocalProviderPage
      providerName="LM Studio"
      endpoint="http://localhost:1234/v1"
      endpointPlaceholder="http://localhost:1234/v1"
      discoveredModels={[]}
      manualModelIds={['manual-1']}
      showOptionalApiKey
      optionalApiKeyValue=""
      onSaveEndpoint={() => {}}
      onRefreshModels={() => {}}
      onAddManualModel={() => {}}
      onRemoveManualModel={() => {}}
      onSaveOptionalApiKey={() => {}}
    />,
  )
}

test('field containers switch border color on hover and restore on mouse out', () => {
  let renderer!: ReturnType<typeof create>
  act(() => {
    renderer = renderPage()
  })

  const getFieldBoxes = () =>
    renderer.root.findAll(
      (n) =>
        n.type === 'box' &&
        typeof n.props.onMouseDown === 'function' &&
        typeof n.props.onMouseOver === 'function' &&
        n.props.style?.borderStyle === 'single' &&
        n.props.style?.paddingLeft === 1 &&
        n.props.style?.flexGrow === 1,
    )

  let fieldBoxes = getFieldBoxes()
  expect(fieldBoxes.length).toBe(3)
  for (const box of fieldBoxes) {
    expect(box.props.style.borderColor).toBe('#444')
  }

  act(() => {
    fieldBoxes[0]!.props.onMouseOver()
  })
  fieldBoxes = getFieldBoxes()
  expect(fieldBoxes[0]!.props.style.borderColor).toBe('#4af')

  act(() => {
    fieldBoxes[0]!.props.onMouseOut()
  })
  fieldBoxes = getFieldBoxes()
  expect(fieldBoxes[0]!.props.style.borderColor).toBe('#444')
})

test('real bordered buttons change label color on hover (save endpoint/add/save key)', () => {
  let renderer!: ReturnType<typeof create>
  act(() => {
    renderer = renderPage()
  })

  const getTextNode = (label: string) =>
    renderer.root.find(
      (n) =>
        n.type === 'text' &&
        (Array.isArray(n.props.children) ? n.props.children.join('') : String(n.props.children ?? '')) === label,
    )

  const getButtonHoverNodeForLabel = (label: string) => {
    const textNode = getTextNode(label)
    let current: any = textNode.parent
    while (current) {
      if (typeof current.props?.onMouseOver === 'function' && typeof current.props?.onMouseOut === 'function') {
        return current
      }
      current = current.parent
    }
    throw new Error(`No hover node found for ${label}`)
  }

  expect(getTextNode('Save endpoint').props.style.fg).toBe('#4af')
  expect(getTextNode('Add').props.style.fg).toBe('#4af')
  expect(getTextNode('Save key').props.style.fg).toBe('#4af')

  const saveEndpointBtn = getButtonHoverNodeForLabel('Save endpoint')
  act(() => {
    saveEndpointBtn.props.onMouseOver()
  })
  expect(getTextNode('Save endpoint').props.style.fg).toBe('#fff')
  act(() => {
    saveEndpointBtn.props.onMouseOut()
  })

  const addBtn = getButtonHoverNodeForLabel('Add')
  act(() => {
    addBtn.props.onMouseOver()
  })
  expect(getTextNode('Add').props.style.fg).toBe('#fff')
  act(() => {
    addBtn.props.onMouseOut()
  })

  const saveKeyBtn = getButtonHoverNodeForLabel('Save key')
  act(() => {
    saveKeyBtn.props.onMouseOver()
  })
  expect(getTextNode('Save key').props.style.fg).toBe('#fff')
})