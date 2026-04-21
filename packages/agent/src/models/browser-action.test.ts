import { describe, expect, test } from 'bun:test'
import { clickModel, navigateModel } from './browser-action'

describe('browser action model', () => {
  test('preserves action-specific streaming label/detail for navigate', () => {
    const started = navigateModel.reduce(navigateModel.initial, { _tag: 'ToolInputStarted' })
    const state = navigateModel.reduce(started, {
      _tag: 'ToolInputFieldChunk',
      field: 'url',
      path: ['url'],
      delta: 'https://example.com',
    })

    expect(state.label).toBe('Navigate')
    expect(state.detail).toBe('https://example.com')
  })

  test('preserves action-specific streaming label/detail for click', () => {
    const started = clickModel.reduce(clickModel.initial, { _tag: 'ToolInputStarted' })
    const state = clickModel.reduce(started, {
      _tag: 'ToolInputReady',
      input: { x: 10, y: 20 },
    })

    expect(state.label).toBe('Click')
    expect(state.detail).toBe('(10, 20)')
  })
})
