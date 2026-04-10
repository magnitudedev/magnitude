import { expect, test } from 'bun:test'
import { computeOptimisticUpdatePreview, findActiveFileStream } from './file-panel-utils'
import type { DisplayState } from '@magnitudedev/agent'

function makeDisplayState(toolHandles: Record<string, any>): DisplayState {
  return {
    status: 'idle',
    messages: [],
    pendingInboundCommunications: [],
    toolHandles,
  } as unknown as DisplayState
}

test('findActiveFileStream returns latest active stream for matching fileWrite path', () => {
  const display = makeDisplayState({
    older: { state: { toolKey: 'fileWrite', path: 'a.ts', phase: 'streaming', body: 'one' } },
    newer: { state: { toolKey: 'fileWrite', path: 'a.ts', phase: 'executing', body: 'two' } },
    other: { state: { toolKey: 'fileWrite', path: 'b.ts', phase: 'streaming', body: 'three' } },
  })

  const match = findActiveFileStream(display, 'a.ts')
  expect(match?.toolCallId).toBe('newer')
  expect(match?.state.toolKey).toBe('fileWrite')
})

test('findActiveFileStream matches fileEdit streaming for exact path only', () => {
  const display = makeDisplayState({
    wrongPath: { state: { toolKey: 'fileEdit', path: 'src/a.ts', phase: 'streaming' } },
    rightPath: { state: { toolKey: 'fileEdit', path: '/tmp/work/src/a.ts', phase: 'streaming' } },
  })

  expect(findActiveFileStream(display, 'src/a.ts')?.toolCallId).toBe('wrongPath')
  expect(findActiveFileStream(display, '/tmp/work/src/a.ts')?.toolCallId).toBe('rightPath')
})

test('computeOptimisticUpdatePreview applies replaceAll edits for live edit preview', () => {
  const preview = computeOptimisticUpdatePreview(
    'hello x\nhello x\n',
    'hello',
    'hi',
    true,
  )

  expect(preview?.content).toBe('hi x\nhi x\n')
  expect(preview?.changedRanges.length).toBe(2)
})