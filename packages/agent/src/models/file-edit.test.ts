import { describe, expect, test } from 'bun:test'
import type { ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import { Option } from 'effect'
import { computeProvisionalEditDiffs, fileEditModel } from './file-edit'

const toolCallId = 'tc1' as ToolCallId
const providerToolCallId = 'tc1' as ProviderToolCallId
const toolName = 'edit'
const toolKey = 'fileEdit'

const baseEvent = {
  toolCallId,
  providerToolCallId,
  toolName,
  toolKey,
}

describe('computeProvisionalEditDiffs', () => {
  test('computes provisional diff for unique match', () => {
    const diffs = computeProvisionalEditDiffs(
      Option.some('a\nb\nold line\nc\nd'),
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
      Option.some('old line\nx\nold line'),
      'old line',
      'new line',
      false,
    )
    expect(diffs).toEqual([])
  })
})

describe('fileEditModel provisional diffs', () => {
  test('populates diffs from base-content emission before completion', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { _tag: 'ToolInputStarted', ...baseEvent })

    let state = fileEditModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'path',
      path: ['path'] as unknown as never,
      delta: 'f.ts',
    })
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'old',
      path: ['old'] as unknown as never,
      delta: 'old',
    })
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'new',
      path: ['new'] as unknown as never,
      delta: 'new',
    })

    expect(state.phase).toBe('streaming')
    expect(state.diffs).toEqual([])

    const withBase = fileEditModel.reduce(state, {
      _tag: 'ToolEmission',
      ...baseEvent,
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
      ...baseEvent,
      result: { _tag: 'Success', output: '' },
    })
    expect(completed.phase).toBe('completed')
    expect(completed.diffs).toHaveLength(1)
    expect(completed.diffs[0]?.addedLines).toEqual(['new'])
  })

  test('diffs appear when baseContent arrives before ToolInputReady', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { _tag: 'ToolInputStarted', ...baseEvent })

    // Simulate path completed (as if ToolInputFieldComplete fired for path)
    let state = fileEditModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'path',
      path: ['path'] as unknown as never,
      delta: 'f.ts',
    })

    // old streaming
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'old',
      path: ['old'] as unknown as never,
      delta: 'old',
    })

    // new streaming
    state = fileEditModel.reduce(state, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'new',
      path: ['new'] as unknown as never,
      delta: 'new',
    })

    // baseContent arrives BEFORE ToolInputReady (the fix enables this)
    const withBase = fileEditModel.reduce(state, {
      _tag: 'ToolEmission',
      ...baseEvent,
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

  test('inputReady uses parsed input values directly', () => {
    const started = fileEditModel.reduce(fileEditModel.initial, { _tag: 'ToolInputStarted', ...baseEvent })

    let state = fileEditModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      toolCallId,
      providerToolCallId,
      field: 'old',
      path: ['old'] as unknown as never,
      delta: 'wrong-old',
    })

    const withInputReady = fileEditModel.reduce(state, {
      _tag: 'ToolInputReady',
      toolCallId,
      providerToolCallId,
    })

    const withExecStarted = fileEditModel.reduce(withInputReady, {
      _tag: 'ToolExecutionStarted',
      ...baseEvent,
      input: {
        path: 'f.ts',
        old: 'old',
        new: 'new',
        replaceAll: false,
      },
      cached: false,
    })

    expect(withExecStarted.phase).toBe('executing')
    expect(withExecStarted.path).toEqual(Option.some('f.ts'))
    expect(withExecStarted.oldText).toBe('old')
    expect(withExecStarted.newText).toBe('new')
    expect(withExecStarted.replaceAll).toBe(false)
    expect(withExecStarted.diffs).toEqual([])

    const withBase = fileEditModel.reduce(withExecStarted, {
      _tag: 'ToolEmission',
      ...baseEvent,
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
