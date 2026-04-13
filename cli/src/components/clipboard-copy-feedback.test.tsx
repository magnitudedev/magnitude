import React from 'react'
import { act, create } from 'react-test-renderer'
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

function deferred() {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

let writeImpl: (() => Promise<void>) | null = null
const originalSpawnSync = Bun.spawnSync

mock.module('../utils/clipboard', () => ({
  writeTextToClipboard: async () => {
    if (writeImpl) return writeImpl()
  },
  readClipboardText: () => null,
}))

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: '#fff',
    muted: '#888',
    primary: '#4af',
    success: '#6f6',
    border: '#444',
    error: '#f66',
    info: '#4af',
    surface: '#111',
    link: '#4af',
    syntax: {},
  }),
}))

mock.module('@opentui/react', () => ({
  useKeyboard: () => {},
  useRenderer: () => ({ clearSelection() {} }),
  useTerminalDimensions: () => ({ width: 80, height: 24 }),
}))

mock.module('../hooks/use-safe-timeout', () => ({
  useSafeTimeout: () => ({ set: setTimeout, clear: clearTimeout }),
}))

mock.module('../hooks/use-mounted-ref', () => ({
  useMountedRef: () => ({ current: true }),
}))

mock.module('../hooks/use-safe-event', () => ({
  useSafeEvent: (fn: unknown) => fn,
}))

const { OAuthOverlay } = await import('./oauth-overlay')
const { CopyButton } = await import('./panel-buttons')
const { BlockRenderer } = await import('../markdown/block-renderer')

function flattenText(node: any): string {
  if (!node) return ''
  if (typeof node === 'string') return node
  if (Array.isArray(node)) return node.map(flattenText).join('')
  return flattenText(node.children)
}

function findCopyBox(root: any) {
  return root.findAllByType('box').find((n: any) => {
    const text = flattenText(n.children)
    return n.props.onMouseDown && (text.includes('[Copy]') || text.includes('[Copy ⧉ ]'))
  })
}

function findCodeCopySurface(root: any) {
  return root.findAllByType('box').find((n: any) => {
    const text = flattenText(n.children)
    return n.props.onMouseDown && text.includes('const x = 1')
  })
}

describe('copy feedback timing', () => {
  beforeEach(() => {
    ;(Bun as any).spawnSync = () => ({ success: true, exitCode: 0, stdout: new Uint8Array(), stderr: new Uint8Array() })
  })

  afterEach(() => {
    ;(Bun as any).spawnSync = originalSpawnSync
    writeImpl = null
  })

  test('oauth copy shows copied only after onCopyUrl resolves', async () => {
    const pending = deferred()
    writeImpl = () => pending.promise

    let renderer!: ReturnType<typeof create>
    await act(async () => {
      renderer = create(
        <OAuthOverlay
          providerName="Provider"
          mode="paste"
          url="https://example.com"
          onCancel={() => {}}
          onCopyUrl={() => pending.promise}
        />,
      )
    })

    const copyButton = findCopyBox(renderer.root)
    expect(copyButton).toBeDefined()

    await act(async () => {
      copyButton!.props.onMouseDown()
      copyButton!.props.onMouseUp()
    })
    expect(flattenText(renderer.toJSON())).not.toContain('[Copied ✓]')

    await act(async () => {
      pending.resolve()
      await Promise.resolve()
    })
    expect(flattenText(renderer.toJSON())).toContain('[Copied ✓]')
  })

  test('panel copy does not show success on write failure', async () => {
    writeImpl = async () => { throw new Error('copy failed') }

    let renderer!: ReturnType<typeof create>
    await act(async () => {
      renderer = create(<CopyButton content="hello" theme={{ success: 'g', foreground: 'w', muted: 'm' }} />)
    })

    const button = findCopyBox(renderer.root)
    await act(async () => {
      button!.props.onMouseDown()
      button!.props.onMouseUp()
      await Promise.resolve()
    })

    const text = flattenText(renderer.toJSON())
    expect(text).toContain('[Copy]')
    expect(text).not.toContain('[✓]')
  })

  test('markdown code copy shows copied only after write resolves', async () => {
    const pending = deferred()
    writeImpl = () => pending.promise

    let renderer!: ReturnType<typeof create>
    await act(async () => {
      renderer = create(
        <BlockRenderer
          blocks={[{
            type: 'code',
            rawCode: 'const x = 1',
            language: 'ts',
            lines: [[{ text: 'const x = 1' }]],
            source: { start: 0, end: 1 },
          } as any]}
          foreground="white"
        />,
      )
    })

    const copySurface = findCodeCopySurface(renderer.root)
    expect(copySurface).toBeDefined()

    await act(async () => {
      copySurface!.props.onMouseOver?.()
      copySurface!.props.onMouseDown({ stopPropagation() {} })
    })
    expect(flattenText(renderer.toJSON())).not.toContain('[Copied ✔]')

    await act(async () => {
      pending.resolve()
      await Promise.resolve()
    })
    expect(flattenText(renderer.toJSON())).toContain('[Copied ✔]')
  })
})
