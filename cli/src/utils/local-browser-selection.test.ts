import { describe, expect, it } from 'vitest'
import { pickLocalBrowserModel } from './local-browser-selection'

describe('pickLocalBrowserModel', () => {
  it('prefers browser-compatible local model when present', () => {
    const selected = pickLocalBrowserModel('lmstudio', [
      { id: 'tinyllama-1.1b', name: 'TinyLlama', status: 'stable' },
      { id: 'qwen3.5:latest', name: 'Qwen', status: 'stable' },
    ])
    expect(selected?.id).toBe('qwen3.5:latest')
  })

  it('falls back to first local model when none are compatible', () => {
    const selected = pickLocalBrowserModel('lmstudio', [
      { id: 'tinyllama-1.1b', name: 'TinyLlama', status: 'stable' },
      { id: 'llama2:13b', name: 'Llama 2', status: 'stable' },
    ])
    expect(selected?.id).toBe('tinyllama-1.1b')
  })

  it('returns null when local inventory is empty', () => {
    const selected = pickLocalBrowserModel('lmstudio', [])
    expect(selected).toBeNull()
  })
})
