import { expect, test } from 'vitest'
import { act } from 'react'
import { testRender } from '@opentui/react/test-utils'
import { createStackedBarCells, StackedBar } from './stacked-bar'

test('uses a foreground block over the following segment background at sub-cell boundaries', () => {
  expect(createStackedBarCells(
    [
      { value: 11, color: 'white' },
      { value: 7, color: 'blue' },
      { value: 6, color: 'orange' },
    ],
    32,
    4,
    { value: 0, color: 'gray', fill: 'shade' },
  )).toEqual([
    { character: '█', foreground: 'white' },
    { character: '▍', foreground: 'white', background: 'blue' },
    { character: '▎', foreground: 'blue', background: 'orange' },
    { character: '░', foreground: 'gray' },
  ])
})

test('renders a fixed-width generic stacked bar', async () => {
  const view = await testRender(
    <StackedBar
      segments={[{ value: 5, color: 'white' }, { value: 2, color: 'blue' }]}
      total={10}
      width={8}
      trackColor="gray"
    />,
    { width: 10, height: 1 },
  )
  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame().trimEnd()).toHaveLength(8)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
