import { describe, expect, test } from 'bun:test'
import { isBrowserCompatible } from '../browser-models'

describe('isBrowserCompatible local provider matching', () => {
  test('matches known-compatible local families (lmstudio/ollama/llama.cpp/openai-compatible-local)', () => {
    expect(isBrowserCompatible('lmstudio', 'google/gemma-3-4b')).toBe(true)
    expect(isBrowserCompatible('ollama', 'qwen3.5:latest')).toBe(true)
    expect(isBrowserCompatible('llama.cpp', 'openai_gpt-oss-20b-q4_k_m.gguf')).toBe(true)
    expect(isBrowserCompatible('openai-compatible-local', 'gpt-4.1-mini')).toBe(true)
  })

  test('keeps unsupported families unsupported for local providers', () => {
    expect(isBrowserCompatible('lmstudio', 'tinyllama-1.1b')).toBe(false)
    expect(isBrowserCompatible('ollama', 'llama2:13b')).toBe(false)
  })

  test('does not change non-local behavior', () => {
    expect(isBrowserCompatible('openai', 'gpt-5.2-codex')).toBe(true)
    expect(isBrowserCompatible('openai', 'gpt-4.1-mini')).toBe(false)
  })
})
