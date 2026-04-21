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
    const started = fileEditModel.reduce(fileEditModel.initial, { _tag: 'ToolInputStarted' })

    let state = fileEditModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      field: 'path',
      path: ['path'] as unknown as never,
      delta: 'f.ts',
    })
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      field: 'oldString',
      path: ['oldString'] as unknown as never,
      delta: 'old',
    })
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      field: 'newString',
      path: ['newString'] as unknown as never,
      delta: 'new',
    })

    expect(state.phase).toBe('streaming')
    expect(state.diffs).toEqual([])

    const withBase = fileEditModel.reduce(state, {
      _tag: 'ToolEmission',
      value: {
        type: 'file_edit_base_content',
        path: 'f.ts',
        baseContent: 'a\nold\nb',
      },
    })

    expect(withBase.phase).toBe('executing')
    expect(withBase.diffs).toHaveLength(1)
    expect(withBase.diffs[0]?.contextBefore).toEqual(['a'])
    expect(withBase.diffs[0]?.removedLines).toEqual(['old'])
    expect(withBase.diffs[0]?.addedLines).toEqual(['new'])
    expect(withBase.diffs[0]?.contextAfter).toEqual(['b'])

    const completed = fileEditModel.reduce(withBase, {
      _tag: 'ToolExecutionEnded',
      result: { _tag: 'Success', output: '', query: null },
    })
    expect(completed.phase).toBe('completed')
    expect(completed.diffs).toHaveLength(1)
    expect(completed.diffs[0]?.addedLines).toEqual(['new'])
  })

  test('inputReady uses parsed input values directly', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { _tag: 'ToolInputStarted' })

    let state = fileEditModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      field: 'oldString',
      path: ['oldString'] as unknown as never,
      delta: 'wrong-old',
    })

    const withInputReady = fileEditModel.reduce(state, {
      _tag: 'ToolInputReady',
      input: {
        path: 'f.ts',
        oldString: 'old',
        newString: 'new',
        replaceAll: false,
      },
    })

    expect(withInputReady.phase).toBe('streaming')
    expect(withInputReady.path).toBe('f.ts')
    expect(withInputReady.oldText).toBe('old')
    expect(withInputReady.newText).toBe('new')
    expect(withInputReady.replaceAll).toBe(false)
    expect(withInputReady.diffs).toEqual([])

    const withBase = fileEditModel.reduce(withInputReady, {
      _tag: 'ToolEmission',
      value: {
        type: 'file_edit_base_content',
        path: 'f.ts',
        baseContent: 'a\nold\nb',
      },
    })

    expect(withBase.diffs).toHaveLength(1)
    expect(withBase.diffs[0]?.contextBefore).toEqual(['a'])
    expect(withBase.diffs[0]?.removedLines).toEqual(['old'])
    expect(withBase.diffs[0]?.addedLines).toEqual(['new'])
    expect(withBase.diffs[0]?.contextAfter).toEqual(['b'])
  })
})
