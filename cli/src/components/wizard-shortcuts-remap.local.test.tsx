import { test, expect, mock } from 'bun:test'
import { act, create } from 'react-test-renderer'
import type { KeyEvent } from '@opentui/core'

const keyboardHandlers: Array<(key: KeyEvent) => void> = []
const dispatchKey = (key: Partial<KeyEvent> & { name: string }) => {
  const handler = keyboardHandlers[keyboardHandlers.length - 1]
  handler?.({ preventDefault: () => {}, ...key } as KeyEvent)
}

let resolveInstall: ((result: { success: boolean; output: string }) => void) | null = null

mock.module('@opentui/react', () => ({
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboardHandlers.push(handler)
  },
}))

mock.module('./single-line-input', () => ({
  SingleLineInput: () => <box />,
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
  }),
}))

mock.module('@magnitudedev/browser-harness', () => ({
  isBrowserInstalled: () => false,
  getBrowserExecutablePath: () => '/tmp/chromium',
  installBrowser: async () => new Promise<{ success: boolean; output: string }>((resolve) => {
    resolveInstall = resolve
  }),
}))

const { ApiKeyOverlay } = await import('./api-key-overlay')
const { AuthMethodOverlay } = await import('./auth-method-overlay')
const { OAuthOverlay } = await import('./oauth-overlay')
const { LocalProviderOverlay } = await import('./local-provider-overlay')
const { BrowserSetupOverlay } = await import('./browser-setup-overlay')

const wizardMode = (onBack: () => void, onSkip: () => void) => ({
  stepLabel: 'Step',
  subtitle: 'Subtitle',
  onBack,
  onSkip,
})

test('api-key overlay wizard shortcuts remap', () => {
  keyboardHandlers.length = 0
  let back = 0
  let skip = 0

  act(() => {
    create(
      <ApiKeyOverlay
        providerName="OpenAI"
        envKeyHint="OPENAI_API_KEY"
        onSubmit={() => {}}
        onCancel={() => {}}
        wizardMode={wizardMode(() => { back += 1 }, () => { skip += 1 })}
      />,
    )
  })

  act(() => dispatchKey({ name: 'escape' }))
  act(() => dispatchKey({ name: 's', ctrl: true }))
  act(() => dispatchKey({ name: 'b' }))

  expect(back).toBe(1)
  expect(skip).toBe(1)
})

test('auth-method overlay wizard shortcuts remap', () => {
  keyboardHandlers.length = 0
  let back = 0
  let skip = 0

  act(() => {
    create(
      <AuthMethodOverlay
        providerName="OpenAI"
        methods={[{ type: 'api-key', label: 'API key', envKeys: ['OPENAI_API_KEY'] } as any]}
        selectedIndex={0}
        onSelectedIndexChange={() => {}}
        onSelect={() => {}}
        onBack={() => {}}
        wizardMode={wizardMode(() => { back += 1 }, () => { skip += 1 })}
      />,
    )
  })

  act(() => dispatchKey({ name: 'escape' }))
  act(() => dispatchKey({ name: 's', ctrl: true }))
  act(() => dispatchKey({ name: 'b' }))

  expect(back).toBe(1)
  expect(skip).toBe(1)
})

test('oauth overlay wizard shortcuts remap (paste mode)', () => {
  keyboardHandlers.length = 0
  let back = 0
  let skip = 0
  const spawnSyncCalls: Array<unknown[]> = []
  const originalSpawnSync = Bun.spawnSync
  ;(Bun as any).spawnSync = (...args: unknown[]) => {
    spawnSyncCalls.push(args)
    return { success: true, exitCode: 0, stdout: new Uint8Array(), stderr: new Uint8Array() }
  }

  try {
    act(() => {
      create(
        <OAuthOverlay
          providerName="OpenAI"
          mode="paste"
          url="https://example.com"
          onCancel={() => {}}
          onCopyUrl={() => {}}
          onSubmitCode={() => {}}
          wizardMode={wizardMode(() => { back += 1 }, () => { skip += 1 })}
        />,
      )
    })

    act(() => dispatchKey({ name: 'escape' }))
    act(() => dispatchKey({ name: 's', ctrl: true }))
    act(() => dispatchKey({ name: 'b' }))

    expect(back).toBe(1)
    expect(skip).toBe(1)

    expect(spawnSyncCalls.length).toBe(1)
    const oauthCall = spawnSyncCalls[0]
    const command = Array.isArray(oauthCall?.[0]) ? oauthCall[0] : []
    expect(command).toContain('https://example.com')
  } finally {
    ;(Bun as any).spawnSync = originalSpawnSync
  }
})

test('local-provider overlay wizard shortcuts remap', () => {
  keyboardHandlers.length = 0
  let back = 0
  let skip = 0

  act(() => {
    create(
      <LocalProviderOverlay
        onSubmit={() => {}}
        onCancel={() => {}}
        wizardMode={wizardMode(() => { back += 1 }, () => { skip += 1 })}
      />,
    )
  })

  act(() => dispatchKey({ name: 'escape' }))
  act(() => dispatchKey({ name: 's', ctrl: true }))
  act(() => dispatchKey({ name: 'b' }))

  expect(back).toBe(1)
  expect(skip).toBe(1)
})

test('browser setup wizard shortcuts remap and disable skip during install', async () => {
  keyboardHandlers.length = 0
  let back = 0
  let skip = 0
  let close = 0

  let renderer!: ReturnType<typeof create>
  await act(async () => {
    renderer = create(
      <BrowserSetupOverlay
        onClose={() => { close += 1 }}
        onResult={() => {}}
        wizardMode={wizardMode(() => { back += 1 }, () => { skip += 1 })}
      />,
    )
  })

  const findSkipNodes = () => {
    const labelNode = renderer.root.findAll((n) => {
      if (n.type !== 'text') return false
      const content = Array.isArray(n.props.children) ? n.props.children.join('') : String(n.props.children ?? '')
      return content === 'Skip (Ctrl+S)'
    })[0]
    const visualBox = labelNode?.parent
    const buttonWrapper = visualBox?.parent
    return { visualBox, buttonWrapper }
  }

  // required state: Esc back, Ctrl+S skip, b no-op, skip button clickable
  const skipBeforeInstall = findSkipNodes()
  expect(skipBeforeInstall.buttonWrapper).toBeDefined()
  expect(skipBeforeInstall.visualBox).toBeDefined()
  expect(skipBeforeInstall.visualBox!.props.style?.opacity).toBeUndefined()
  act(() => {
    skipBeforeInstall.buttonWrapper!.props.onMouseDown()
    skipBeforeInstall.buttonWrapper!.props.onMouseUp()
  })
  expect(skip).toBe(1)

  act(() => dispatchKey({ name: 'escape' }))
  act(() => dispatchKey({ name: 's', ctrl: true }))
  act(() => dispatchKey({ name: 'b' }))
  expect(back).toBe(1)
  expect(skip).toBe(2)
  expect(close).toBe(0)

  // installing state disables both Esc/Ctrl+S and disables skip button click path
  await act(async () => {
    dispatchKey({ name: 'enter' })
  })

  const skipDuringInstall = findSkipNodes()
  expect(skipDuringInstall.buttonWrapper).toBeDefined()
  expect(skipDuringInstall.visualBox).toBeDefined()
  expect(skipDuringInstall.visualBox!.props.style?.opacity).toBe(0.6)
  act(() => {
    skipDuringInstall.buttonWrapper!.props.onMouseDown()
    skipDuringInstall.buttonWrapper!.props.onMouseUp()
  })

  act(() => dispatchKey({ name: 'escape' }))
  act(() => dispatchKey({ name: 's', ctrl: true }))
  expect(back).toBe(1)
  expect(skip).toBe(2)

  await act(async () => {
    resolveInstall?.({ success: true, output: '' })
  })
})