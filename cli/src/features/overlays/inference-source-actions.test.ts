import { describe, expect, test } from 'vitest'
import {
  getInferenceSourceAction,
  INFERENCE_SOURCE_ACTIONS,
} from './inference-source-actions'

describe('settings inference source actions', () => {
  test('exposes local model management', () => {
    expect(INFERENCE_SOURCE_ACTIONS.local.label).toBe('Manage local models')
  })

  test('routes the local shortcut and rejects unrelated keys', () => {
    expect(getInferenceSourceAction('l')).toBe('local')
    expect(getInferenceSourceAction('c')).toBeNull()
    expect(getInferenceSourceAction('x')).toBeNull()
  })
})
