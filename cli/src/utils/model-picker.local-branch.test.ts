import { describe, expect, it } from 'vitest'
import type { ProviderDefinition } from '@magnitudedev/providers'
import { resolveSlotDefaultSelection } from './model-picker'

function makeProvider(overrides: Partial<ProviderDefinition> & Pick<ProviderDefinition, 'id' | 'name'>): ProviderDefinition {
  return {
    id: overrides.id,
    name: overrides.name,
    envVars: [],
    auth: [{ type: 'none' }],
    supportsModelListing: true,
    models: [],
    ...overrides,
  }
}

describe('resolveSlotDefaultSelection (local browser slot behavior)', () => {
  it('local browser slot chooses compatible local model when available', () => {
    const provider = makeProvider({
      id: 'lmstudio',
      name: 'LM Studio',
      models: [
        { id: 'tinyllama-1.1b', name: 'TinyLlama', status: 'stable' },
        { id: 'qwen3.5:latest', name: 'Qwen 3.5', status: 'stable' },
      ],
    })

    const result = resolveSlotDefaultSelection({
      allProviders: [provider],
      connectedProviderIds: new Set(['lmstudio']),
      slot: 'browser',
      preferredProviderId: 'lmstudio',
    })

    expect(result).toEqual({ providerId: 'lmstudio', modelId: 'qwen3.5:latest' })
  })

  it('local browser slot falls back to first local model when none are compatible', () => {
    const provider = makeProvider({
      id: 'ollama',
      name: 'Ollama',
      models: [
        { id: 'tinyllama-1.1b', name: 'TinyLlama', status: 'stable' },
        { id: 'llama2:13b', name: 'Llama 2', status: 'stable' },
      ],
    })

    const result = resolveSlotDefaultSelection({
      allProviders: [provider],
      connectedProviderIds: new Set(['ollama']),
      slot: 'browser',
      preferredProviderId: 'ollama',
    })

    expect(result).toEqual({ providerId: 'ollama', modelId: 'tinyllama-1.1b' })
  })

  it('local browser slot returns null only when local inventory is empty', () => {
    const provider = makeProvider({
      id: 'llama.cpp',
      name: 'llama.cpp',
      models: [],
    })

    const result = resolveSlotDefaultSelection({
      allProviders: [provider],
      connectedProviderIds: new Set(['llama.cpp']),
      slot: 'browser',
      preferredProviderId: 'llama.cpp',
    })

    expect(result).toBeNull()
  })

  it('non-local behavior remains unchanged (falls back to first available when no browser-compatible model exists)', () => {
    const provider = makeProvider({
      id: 'openai',
      name: 'OpenAI',
      models: [
        { id: 'custom-model-a', name: 'Custom A', status: 'stable' },
        { id: 'custom-model-b', name: 'Custom B', status: 'stable' },
      ],
    })

    const result = resolveSlotDefaultSelection({
      allProviders: [provider],
      connectedProviderIds: new Set(['openai']),
      slot: 'browser',
      preferredProviderId: 'openai',
    })

    expect(result).toEqual({ providerId: 'openai', modelId: 'custom-model-a' })
  })
})
