import { act, useState } from 'react'
import { testRender } from '@opentui/react/test-utils'
import type { ScrollBoxRenderable } from '@opentui/core'
import { expect, test } from 'vitest'
import { ChatScrollbox } from './scrollbox'

test('root chat scrollbox uses native overflow-only tail following', async () => {
  const source = await Bun.file(new URL('./scrollbox.tsx', import.meta.url)).text()

  expect(source).toContain('stickyScroll={true}')
  expect(source).toContain('stickyStart="bottom"')
  expect(source).toContain("justifyContent: 'flex-start'")
  expect(source).not.toContain("justifyContent: 'flex-end'")
})

test('the live activity rail is outside history and directly precedes the composer', async () => {
  const timelineSource = await Bun.file(new URL('./container.tsx', import.meta.url)).text()
  const appSource = await Bun.file(new URL('../../app.tsx', import.meta.url)).text()

  expect(timelineSource).not.toContain('<ActivityRailContainer')
  expect(appSource.indexOf('{activityRail}', appSource.indexOf('<TaskListContainer')))
    .toBeGreaterThan(appSource.indexOf('<TaskListContainer'))
  expect(appSource.indexOf('{activityRail}', appSource.indexOf('<TaskListContainer')))
    .toBeLessThan(appSource.indexOf('<ComposerContainer'))
})

test('appended rows stay in place until they overflow the viewport', async () => {
  let surface: ScrollBoxRenderable | null = null
  let addRows: ((count: number) => void) | undefined

  function Harness() {
    const [rowCount, setRowCount] = useState(1)
    addRows = (count) => setRowCount((current) => current + count)
    return (
      <ChatScrollbox scrollRef={(value) => { surface = value }}>
        {Array.from({ length: rowCount }, (_, index) => (
          <box key={index} style={{ height: 1, flexShrink: 0 }}>
            <text>{`row ${index}`}</text>
          </box>
        ))}
      </ChatScrollbox>
    )
  }

  const view = await testRender(<Harness />, { width: 50, height: 8 })
  try {
    await act(view.renderOnce)
    expect(surface).not.toBeNull()
    expect(surface!.scrollTop).toBe(0)

    const availableRows = Math.max(0, surface!.viewport.height - surface!.scrollHeight)
    await act(async () => addRows?.(availableRows))
    await act(view.renderOnce)
    expect(surface!.scrollHeight).toBeLessThanOrEqual(surface!.viewport.height)
    expect(surface!.scrollTop).toBe(0)

    await act(async () => addRows?.(1))
    await act(view.renderOnce)
    expect(surface!.scrollTop).toBe(surface!.scrollHeight - surface!.viewport.height)

    await act(async () => addRows?.(2))
    await act(view.renderOnce)
    surface!.scrollTo(0)
    await act(async () => addRows?.(1))
    await act(view.renderOnce)
    expect(surface!.scrollTop).toBe(0)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
