import { describe, expect, test } from 'bun:test'
import {
  localProviderAddedModelToast,
  localProviderSavedApiKeyToast,
  localProviderSavedEndpointToast,
} from './utils/local-provider-toast-messages'

describe('local provider success toast messages', () => {
  test('save endpoint success toast content', () => {
    expect(localProviderSavedEndpointToast('LM Studio')).toBe('Saved endpoint for LM Studio')
  })

  test('add manual model success toast content', () => {
    expect(localProviderAddedModelToast('qwen2.5-coder')).toBe('Added model qwen2.5-coder')
  })

  test('save key success toast content', () => {
    expect(localProviderSavedApiKeyToast('LM Studio')).toBe('Saved API key for LM Studio')
  })
})
