import { describe, expect, test } from 'bun:test'
import { clickModel, navigateModel } from './browser-action'

describe('browser action model', () => {
  test('preserves action-specific streaming label/detail for navigate', () => {
    const started = navigateModel.reduce(navigateModel.initial, { type: 'started' })
    const state = navigateModel.reduce(started, {
      type: 'inputUpdated',
      changed: 'field',
      name: 'url',
      streaming: { fields: { url: 'https://example.com' }, body: '', children: {} },
    })

    expect(state.label).toBe('Navigate')
    expect(state.detail).toBe('https://example.com')
  })

  test('preserves action-specific streaming label/detail for click', () => {
    const started = clickModel.reduce(clickModel.initial, { type: 'started' })
    const state = clickModel.reduce(started, {
      type: 'inputReady',
      input: { x: 10, y: 20 },
      streaming: { fields: { x: '10', y: '20' }, body: '', children: {} },
    })

    expect(state.label).toBe('Click')
    expect(state.detail).toBe('(10, 20)')
  })
})
