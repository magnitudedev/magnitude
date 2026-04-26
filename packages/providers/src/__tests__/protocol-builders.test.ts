import { afterEach, describe, expect, it, vi } from 'vitest'
import { __testOnly_buildProviderOptions } from '../client-registry-builder'
import type { OpenAIGenericOptions } from '../protocol/types'

describe('Protocol builders', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  // -------------------------------------------------------------------------
  // Fireworks AI — openai-generic with grammar support
  // -------------------------------------------------------------------------
  describe('fireworks-ai', () => {
    it('builds options with grammar via response_format', () => {
      vi.stubEnv('FIREWORKS_API_KEY', 'fw-key')
      const opts = __testOnly_buildProviderOptions('fireworks-ai', 'glm-5.1', null, undefined, undefined, 'root ::= "a"')
      expect(opts).toEqual(expect.objectContaining({
        model: 'glm-5.1',
        api_key: 'fw-key',
        base_url: 'https://api.fireworks.ai/inference/v1',
        response_format: { type: 'grammar', grammar: 'root ::= "a"' },
        reasoning_effort: 'none',
      }))
    })

    it('uses reasoning_effort=low for minimax models', () => {
      vi.stubEnv('FIREWORKS_API_KEY', 'fw-key')
      const opts = __testOnly_buildProviderOptions('fireworks-ai', 'minimax-m2.7', null)
      expect(opts).toEqual(expect.objectContaining({ reasoning_effort: 'low' }))
    })

    it('returns undefined when no auth available', () => {
      vi.stubEnv('FIREWORKS_API_KEY', '')
      const opts = __testOnly_buildProviderOptions('fireworks-ai', 'glm-5.1', null)
      expect(opts).toBeUndefined()
    })

    it('uses explicit api auth', () => {
      const opts = __testOnly_buildProviderOptions('fireworks-ai', 'glm-5.1', { type: 'api', key: 'explicit-key' })
      expect(opts).toEqual(expect.objectContaining({ api_key: 'explicit-key' }))
    })
  })

  // -------------------------------------------------------------------------
  // Anthropic — anthropic protocol
  // -------------------------------------------------------------------------
  describe('anthropic', () => {
    it('builds options with api key auth', () => {
      const opts = __testOnly_buildProviderOptions('anthropic', 'claude-opus-4-5-20251101', { type: 'api', key: 'ant-key' }, undefined, ['STOP'])
      expect(opts).toEqual(expect.objectContaining({
        model: 'claude-opus-4-5-20251101',
        api_key: 'ant-key',
        stop_sequences: ['STOP'],
      }))
    })

    it('builds oauth headers', () => {
      const opts = __testOnly_buildProviderOptions('anthropic', 'claude-opus-4-5-20251101', { type: 'oauth', oauthMethod: 'oauth-pkce' as const, accessToken: 'tok', refreshToken: 'ref', expiresAt: 9999999999 })
      expect(opts?.headers).toBeDefined()
      expect(opts?.headers?.['Authorization']).toBe('Bearer tok')
      expect(opts?.api_key).toBeUndefined()
    })

    it('falls back to ANTHROPIC_API_KEY env', () => {
      vi.stubEnv('ANTHROPIC_API_KEY', 'env-ant')
      const opts = __testOnly_buildProviderOptions('anthropic', 'claude-opus-4-5-20251101', null)
      expect(opts).toEqual(expect.objectContaining({ api_key: 'env-ant' }))
    })

    it('returns undefined when no auth', () => {
      vi.stubEnv('ANTHROPIC_API_KEY', '')
      const opts = __testOnly_buildProviderOptions('anthropic', 'claude-opus-4-5-20251101', null)
      expect(opts).toBeUndefined()
    })
  })

  // -------------------------------------------------------------------------
  // llama.cpp — grammar uses `grammar` field, not `response_format`
  // -------------------------------------------------------------------------
  describe('llama.cpp', () => {
    it('uses grammar field directly', () => {
      const opts = __testOnly_buildProviderOptions('llama.cpp', 'llama3', null, undefined, undefined, 'root ::= "x"')
      expect(opts).toEqual(expect.objectContaining({
        grammar: 'root ::= "x"',
      }))
      expect((opts as OpenAIGenericOptions | undefined)?.response_format).toBeUndefined()
    })

    it('runs without auth (local provider)', () => {
      const opts = __testOnly_buildProviderOptions('llama.cpp', 'llama3', null)
      expect(opts).toBeDefined()
      expect(opts?.model).toBe('llama3')
    })
  })

  // -------------------------------------------------------------------------
  // OpenRouter — no grammar support
  // -------------------------------------------------------------------------
  describe('openrouter', () => {
    it('builds options with api key', () => {
      vi.stubEnv('OPENROUTER_API_KEY', 'or-key')
      const opts = __testOnly_buildProviderOptions('openrouter', 'meta-llama/llama-3.1-8b-instruct', null, undefined, ['stop'])
      expect(opts).toEqual(expect.objectContaining({
        api_key: 'or-key',
        base_url: 'https://openrouter.ai/api/v1',
        stop: ['stop'],
      }))
    })

    it('does not set response_format even when grammar provided', () => {
      vi.stubEnv('OPENROUTER_API_KEY', 'or-key')
      const opts = __testOnly_buildProviderOptions('openrouter', 'meta-llama/llama-3.1-8b-instruct', null, undefined, undefined, 'root ::= "a"')
      // openrouter has no grammar capability — grammar arg is ignored
      expect((opts as OpenAIGenericOptions | undefined)?.response_format).toBeUndefined()
      expect((opts as OpenAIGenericOptions | undefined)?.grammar).toBeUndefined()
    })
  })

  // -------------------------------------------------------------------------
  // Ollama — local, no auth required
  // -------------------------------------------------------------------------
  describe('ollama', () => {
    it('builds options without auth', () => {
      const opts = __testOnly_buildProviderOptions('ollama', 'llama3.2', null)
      expect(opts).toEqual(expect.objectContaining({
        model: 'llama3.2',
        base_url: 'http://localhost:11434/v1',
      }))
      expect(opts?.api_key).toBeUndefined()
    })
  })

  // -------------------------------------------------------------------------
  // Z.AI — thinking disabled
  // -------------------------------------------------------------------------
  describe('zai', () => {
    it('disables thinking', () => {
      vi.stubEnv('ZHIPU_API_KEY', 'zai-key')
      const opts = __testOnly_buildProviderOptions('zai', 'glm-4-plus', null)
      expect(opts).toEqual(expect.objectContaining({
        thinking: { type: 'disabled' },
      }))
    })
  })

  // -------------------------------------------------------------------------
  // DeepSeek — openai-generic, thinking disabled
  // -------------------------------------------------------------------------
  describe('deepseek', () => {
    it('builds options with api key auth', () => {
      vi.stubEnv('DEEPSEEK_API_KEY', 'ds-key')
      const opts = __testOnly_buildProviderOptions('deepseek', 'deepseek-v4-pro', null, 1000, ['stop'])
      expect(opts).toEqual(expect.objectContaining({
        model: 'deepseek-v4-pro',
        api_key: 'ds-key',
        base_url: 'https://api.deepseek.com/v1',
        stop: ['stop'],
        thinking: { type: 'disabled' },
        stream_options: { include_usage: true },
      }))
    })

    it('disables thinking for all models', () => {
      vi.stubEnv('DEEPSEEK_API_KEY', 'ds-key')
      const optsPro = __testOnly_buildProviderOptions('deepseek', 'deepseek-v4-pro', null)
      const optsFlash = __testOnly_buildProviderOptions('deepseek', 'deepseek-v4-flash', null)
      expect(optsPro).toEqual(expect.objectContaining({ thinking: { type: 'disabled' } }))
      expect(optsFlash).toEqual(expect.objectContaining({ thinking: { type: 'disabled' } }))
    })

    it('returns undefined when no auth available', () => {
      vi.stubEnv('DEEPSEEK_API_KEY', '')
      const opts = __testOnly_buildProviderOptions('deepseek', 'deepseek-v4-pro', null)
      expect(opts).toBeUndefined()
    })

    it('uses explicit api auth', () => {
      const opts = __testOnly_buildProviderOptions('deepseek', 'deepseek-v4-pro', { type: 'api', key: 'explicit-key' })
      expect(opts).toEqual(expect.objectContaining({ api_key: 'explicit-key' }))
    })
  })
})
