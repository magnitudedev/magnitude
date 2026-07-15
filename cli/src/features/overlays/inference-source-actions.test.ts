import { describe, expect, test } from 'vitest'
import {
  getInferenceSourceAction,
  INFERENCE_SOURCE_ACTIONS,
} from './inference-source-actions'

describe('settings inference source actions', () => {
  test('exposes independent local and cloud management entry points', () => {
    expect(INFERENCE_SOURCE_ACTIONS.local.label).toBe('Manage local models')
    expect(INFERENCE_SOURCE_ACTIONS.cloud.label).toBe('Configure Cloud fallback')
  })

  test('routes the documented shortcuts to their domains', () => {
    expect(getInferenceSourceAction('l')).toBe('local')
    expect(getInferenceSourceAction('c')).toBe('cloud')
    expect(getInferenceSourceAction('x')).toBeNull()
  })
})
