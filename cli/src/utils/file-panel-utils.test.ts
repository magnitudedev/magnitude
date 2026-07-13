import { expect, test } from 'bun:test'
import { computeOptimisticUpdatePreview, findActiveFileStream } from './file-panel-utils'

function makeToolHandles(toolHandles: Record<string, any>) {
  return toolHandles
}

test('findActiveFileStream returns latest active stream for matching fileWrite path', () => {
  const toolHandles = makeToolHandles({
    older: { toolKey: 'fileWrite', state: { phase: 'streaming', path: 'a.ts', body: 'one', charCount: 3, lineCount: 1 } },
    newer: { toolKey: 'fileWrite', state: { phase: 'executing', path: 'a.ts', body: 'two', charCount: 3, lineCount: 1 } },
    other: { toolKey: 'fileWrite', state: { phase: 'streaming', path: 'b.ts', body: 'three', charCount: 5, lineCount: 1 } },
  })

  const match = findActiveFileStream(toolHandles, 'a.ts')
  expect(match?.toolCallId).toBe('newer')
  expect(match?.state).toMatchObject({ body: 'two' })
})

test('findActiveFileStream matches fileEdit streaming for exact path only', () => {
  const toolHandles = makeToolHandles({
    wrongPath: { toolKey: 'fileEdit', state: { phase: 'streaming', path: 'src/a.ts', oldText: '', newText: '', replaceAll: false, diffs: [] } },
    rightPath: { toolKey: 'fileEdit', state: { phase: 'streaming', path: '/tmp/work/src/a.ts', oldText: '', newText: '', replaceAll: false, diffs: [] } },
  })

  expect(findActiveFileStream(toolHandles, 'src/a.ts')?.toolCallId).toBe('wrongPath')
  expect(findActiveFileStream(toolHandles, '/tmp/work/src/a.ts')?.toolCallId).toBe('rightPath')
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