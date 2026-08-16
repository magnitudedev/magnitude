import { act } from 'react'
import { testRender } from '@opentui/react/test-utils'
import { expect, test } from 'vitest'
import { useTerminalWidth } from './use-terminal-width'

function WidthProbe() {
  return <text>{String(useTerminalWidth())}</text>
}

test('tracks the OpenTUI renderer width across terminal resizes', async () => {
  const view = await testRender(<WidthProbe />, { width: 42, height: 4 })

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame().split('\n')[0]?.trim()).toBe('42')

    await act(async () => view.resize(17, 4))
    await act(view.renderOnce)
    expect(view.captureCharFrame().split('\n')[0]?.trim()).toBe('17')
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
