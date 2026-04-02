import { test, expect, mock } from 'bun:test'
import { act, create } from 'react-test-renderer'
import React from 'react'

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
const { SingleLineInput } = await import('./single-line-input')

test('provider page supports save -> edit -> save for optional API key', () => {
  function Harness() {
    const [savedKey, setSavedKey] = React.useState('')
    const [draft, setDraft] = React.useState('')

    return (
      <LocalProviderPage
        providerName="LM Studio"
        endpoint="http://localhost:1234/v1"
        endpointPlaceholder="http://localhost:1234/v1"
        discoveredModels={[]}
        manualModelIds={[]}
        showOptionalApiKey
        hasSavedOptionalApiKey={savedKey.trim().length > 0}
        optionalApiKeyValue={savedKey}
        onSaveEndpoint={() => {}}
        onRefreshModels={() => {}}
        onAddManualModel={() => {}}
        onRemoveManualModel={() => {}}
        onSaveOptionalApiKey={(value) => {
          setDraft(value)
          setSavedKey(value.trim())
        }}
      />
    )
  }

  let renderer!: ReturnType<typeof create>
  act(() => {
    renderer = create(<Harness />)
  })

  const findApiInput = () =>
    renderer.root
      .findAllByType(SingleLineInput)
      .find((node) => node.props.placeholder === 'Leave empty if not required')

  const clickButtonByLabel = (label: string) => {
    const clickableNode = renderer.root.findAll((node) => {
      if (typeof node.props?.onMouseDown !== 'function' || typeof node.props?.onMouseUp !== 'function') return false
      const texts = node.findAll(
        (n) => n.type === 'text' && (Array.isArray(n.props.children) ? n.props.children.join('') : String(n.props.children ?? '')).includes(label),
      )
      return texts.length > 0
    })[0]
    expect(clickableNode).toBeDefined()
    act(() => {
      clickableNode!.props.onMouseDown()
      clickableNode!.props.onMouseUp()
    })
  }

  const hasLabel = (label: string) =>
    renderer.root.findAll(
      (n) => n.type === 'text' && (Array.isArray(n.props.children) ? n.props.children.join('') : String(n.props.children ?? '')) === label,
    ).length > 0

  const apiInput1 = findApiInput()
  expect(apiInput1).toBeDefined()
  act(() => {
    apiInput1!.props.onChange('first-key')
  })
  clickButtonByLabel('Save key')

  expect(hasLabel('Delete key')).toBe(true)

  const apiInput2 = findApiInput()
  expect(apiInput2).toBeDefined()
  act(() => {
    apiInput2!.props.onChange('updated-key')
  })

  expect(hasLabel('Save key')).toBe(true)
  clickButtonByLabel('Save key')

  expect(hasLabel('Delete key')).toBe(true)
})