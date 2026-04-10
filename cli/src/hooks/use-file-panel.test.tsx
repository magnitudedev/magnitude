import { beforeAll, expect, mock, test } from 'bun:test'
import React from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'

let useFilePanel: typeof import('./use-file-panel').useFilePanel

beforeAll(async () => {
  mock.restore()
  ;({ useFilePanel } = await import('./use-file-panel'))
})

type PanelResult = ReturnType<typeof useFilePanel>

function Harness({ onState }: { onState: (left: PanelResult, right: PanelResult) => void }) {
  const left = useFilePanel({ display: null, workspacePath: null, projectRoot: '/tmp/project' })
  const right = useFilePanel({ display: null, workspacePath: null, projectRoot: '/tmp/project' })
  onState(left, right)
  return null
}

test('useFilePanel instances are independent', () => {
  let left!: PanelResult
  let right!: PanelResult
  let renderer: ReactTestRenderer

  act(() => {
    renderer = create(<Harness onState={(l, r) => { left = l; right = r }} />)
  })

  act(() => {
    left.openFile('left.md')
  })

  expect(left.selectedFile?.path).toBe('left.md')
  expect(right.selectedFile).toBeNull()

  renderer!.unmount()
})

test('useFilePanel state resets after unmount/remount lifecycle', () => {
  let left!: PanelResult
  let right!: PanelResult
  let renderer: ReactTestRenderer

  act(() => {
    renderer = create(<Harness onState={(l, r) => { left = l; right = r }} />)
  })

  act(() => {
    left.openFile('overlay.md')
  })
  expect(left.selectedFile?.path).toBe('overlay.md')

  act(() => {
    renderer!.unmount()
  })

  act(() => {
    renderer = create(<Harness onState={(l, r) => { left = l; right = r }} />)
  })

  expect(left.selectedFile).toBeNull()
  expect(right.selectedFile).toBeNull()

  renderer!.unmount()
})