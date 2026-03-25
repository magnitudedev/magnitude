import { describe, expect, test } from 'bun:test'
import { computeProvisionalEditDiffs, fileEditModel } from './file-edit'

describe('computeProvisionalEditDiffs', () => {
  test('computes provisional diff for unique match', () => {
    const diffs = computeProvisionalEditDiffs(
      'a\nb\nold line\nc\nd',
      'old line',
      'new line',
      false,
    )

    expect(diffs).toHaveLength(1)
    expect(diffs[0]?.contextBefore).toEqual(['a', 'b'])
    expect(diffs[0]?.removedLines).toEqual(['old line'])
    expect(diffs[0]?.addedLines).toEqual(['new line'])
    expect(diffs[0]?.contextAfter).toEqual(['c', 'd'])
  })

  test('returns empty for ambiguous match', () => {
    const diffs = computeProvisionalEditDiffs(
      'old line\nx\nold line',
      'old line',
      'new line',
      false,
    )
    expect(diffs).toEqual([])
  })
})

describe('fileEditModel provisional diffs', () => {
  test('populates diffs from base-content emission before completion', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { type: 'started' })

    const withInputs = fileEditModel.reduce(started, {
      type: 'inputUpdated',
      streaming: {
        path: { value: 'f.ts', isFinal: true },
        oldString: { value: 'old', isFinal: true },
        newString: { value: 'new', isFinal: true },
        replaceAll: { value: 'false', isFinal: false },
      },
      changed: 'field',
      name: 'oldString',
    })

    expect(withInputs.phase).toBe('streaming')
    expect(withInputs.diffs).toEqual([])

    const withBase = fileEditModel.reduce(withInputs, {
      type: 'emission',
      value: {
        type: 'file_edit_base_content',
        path: 'f.ts',
        baseContent: 'a\nold\nb',
      },
    } as any)

    expect(withBase.phase).toBe('executing')
    expect(withBase.diffs).toHaveLength(1)
    expect(withBase.diffs[0]?.contextBefore).toEqual(['a'])
    expect(withBase.diffs[0]?.removedLines).toEqual(['old'])
    expect(withBase.diffs[0]?.addedLines).toEqual(['new'])
    expect(withBase.diffs[0]?.contextAfter).toEqual(['b'])

    const completed = fileEditModel.reduce(withBase, {
      type: 'completed',
      result: '',
      output: undefined,
    } as any)
    expect(completed.phase).toBe('completed')
    expect(completed.diffs).toHaveLength(1)
    expect(completed.diffs[0]?.addedLines).toEqual(['new'])
  })

  test('inputReady uses parsed input values directly', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { type: 'started' })

    const withStreaming = fileEditModel.reduce(started, {
      type: 'inputUpdated',
      streaming: {
        path: { value: 'wrong.ts', isFinal: true },
        oldString: { value: 'wrong-old', isFinal: true },
        newString: { value: 'wrong-new', isFinal: true },
        replaceAll: { value: 'true', isFinal: false },
      },
      changed: 'field',
      name: 'oldString',
    } as any)

    const withInputReady = fileEditModel.reduce(withStreaming, {
      type: 'inputReady',
      input: {
        path: 'f.ts',
        oldString: 'old',
        newString: 'new',
        replaceAll: false,
      },
      streaming: {
        path: { value: 'still-wrong.ts', isFinal: true },
        oldString: { value: 'still-wrong-old', isFinal: true },
        newString: { value: 'still-wrong-new', isFinal: true },
        replaceAll: { value: 'true', isFinal: false },
      },
    } as any)

    expect(withInputReady.phase).toBe('streaming')
    expect(withInputReady.path).toBe('f.ts')
    expect(withInputReady.oldText).toBe('old')
    expect(withInputReady.newText).toBe('new')
    expect(withInputReady.replaceAll).toBe(false)
    expect(withInputReady.diffs).toEqual([])

    const withBase = fileEditModel.reduce(withInputReady, {
      type: 'emission',
      value: {
        type: 'file_edit_base_content',
        path: 'f.ts',
        baseContent: 'a\nold\nb',
      },
    } as any)

    expect(withBase.diffs).toHaveLength(1)
    expect(withBase.diffs[0]?.contextBefore).toEqual(['a'])
    expect(withBase.diffs[0]?.removedLines).toEqual(['old'])
    expect(withBase.diffs[0]?.addedLines).toEqual(['new'])
    expect(withBase.diffs[0]?.contextAfter).toEqual(['b'])
  })
})
