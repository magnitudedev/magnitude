import { describe, expect, it } from 'vitest'
import { tryResolveCanonicalModelId } from '../registry'

describe('tryResolveCanonicalModelId', () => {
  it('resolves direct canonical ID matches', () => {
    expect(tryResolveCanonicalModelId('kimi-k2.5')).toBe('kimi-k2.5')
    expect(tryResolveCanonicalModelId('glm-5.1')).toBe('glm-5.1')
    expect(tryResolveCanonicalModelId('minimax-m2.5')).toBe('minimax-m2.5')
  })

  it('resolves namespace-stripped IDs', () => {
    expect(tryResolveCanonicalModelId('moonshotai/kimi-k2.5')).toBe('kimi-k2.5')
    expect(tryResolveCanonicalModelId('moonshotai/kimi-k2.6')).toBe('kimi-k2.6')
    expect(tryResolveCanonicalModelId('zai/glm-5.1')).toBe('glm-5.1')
  })

  it('returns null for unknown IDs', () => {
    expect(tryResolveCanonicalModelId('claude-opus-4.7')).toBeNull()
    expect(tryResolveCanonicalModelId('gpt-5.4')).toBeNull()
    expect(tryResolveCanonicalModelId('random-model')).toBeNull()
  })

  it('returns null for namespace-stripped unknown IDs', () => {
    expect(tryResolveCanonicalModelId('anthropic/unknown-model')).toBeNull()
    expect(tryResolveCanonicalModelId('openai/gpt-4')).toBeNull()
  })
})

