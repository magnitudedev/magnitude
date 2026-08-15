import { act } from 'react'
import { expect, test, vi } from 'vitest'
import { testRender } from '@opentui/react/test-utils'
import type { Block } from './blocks'
import { defaultCliThemes } from '../utils/theme'

vi.mock('../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
}))

const { BlockRenderer } = await import('./block-renderer')

const inlineFileBlock: Block = {
  type: 'paragraph',
  source: { start: 0, end: 22 },
  content: [
    { text: 'abcdefghij ' },
    {
      text: 'wrapped',
      fileRef: { path: '/tmp/wrapped.md', section: 'intro' },
    },
    { text: ' tail' },
  ],
}

test('wrapped inline file links preserve hover and click behavior', async () => {
  const onOpenFile = vi.fn()
  const view = await testRender(
    <BlockRenderer
      blocks={[inlineFileBlock]}
      foreground="#ffffff"
      contentWidth={12}
      onOpenFile={onOpenFile}
    />,
    { width: 12, height: 4 },
  )

  try {
    await act(view.renderOnce)
    const lines = view.captureCharFrame().split('\n')
    const row = lines.findIndex((line) => line.includes('wrapped'))
    const column = lines[row]!.indexOf('wrapped') + 2
    const pointers: string[] = []
    view.renderer.setMousePointer = ((pointer: string) => {
      pointers.push(pointer)
    }) as typeof view.renderer.setMousePointer

    await act(async () => view.mockMouse.moveTo(column, row))
    await act(view.renderOnce)
    expect(pointers).toContain('pointer')

    await act(async () => view.mockMouse.click(column, row))
    expect(onOpenFile).toHaveBeenCalledWith('/tmp/wrapped.md', 'intro')

    await act(async () => view.mockMouse.moveTo(11, 3))
    expect(pointers.at(-1)).toBe('default')
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test('nested inline file links use terminal-global mouse coordinates', async () => {
  const onOpenFile = vi.fn()
  const view = await testRender(
    <box style={{ paddingLeft: 5, paddingTop: 2, flexDirection: 'column' }}>
      <box style={{ marginLeft: 3, marginTop: 1, width: 12 }}>
        <BlockRenderer
          blocks={[inlineFileBlock]}
          foreground="#ffffff"
          contentWidth={12}
          onOpenFile={onOpenFile}
        />
      </box>
    </box>,
    { width: 30, height: 10 },
  )

  try {
    await act(view.renderOnce)
    const lines = view.captureCharFrame().split('\n')
    const row = lines.findIndex((line) => line.includes('wrapped'))
    const column = lines[row]!.indexOf('wrapped') + 2
    expect(row).toBeGreaterThan(2)
    expect(column).toBeGreaterThan(5)

    await act(async () => view.mockMouse.click(column, row))
    expect(onOpenFile).toHaveBeenCalledWith('/tmp/wrapped.md', 'intro')
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
