/**
 * Auth resolution — maps AuthStrategy + AuthInfo to typed option partials.
 *
 * Returns undefined when no valid auth is available (caller should skip registration).
 */

import type { AuthInfo } from '../types'
import type { AuthStrategy, AnthropicOptions, OpenAIOptions, OpenAIGenericOptions } from './types'

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

export function resolveAnthropicAuth(
  strategy: AuthStrategy,
  auth: AuthInfo | null,
): Partial<AnthropicOptions> | undefined {
  switch (strategy.type) {
    case 'api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      // Fall back to env vars
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return undefined
    }
    case 'oauth-anthropic': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') {
        return {
          headers: {
            'Authorization': `Bearer ${auth.accessToken}`,
            'anthropic-beta': 'oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14',
            'X-API-Key': '',
          },
        }
      }
      // Fall back to env var
      const envKey = process.env.ANTHROPIC_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'oauth-as-api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      return undefined
    }
    case 'oauth-openai': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      const envKey = process.env.OPENAI_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'none':
      return {}
    case 'local-optional': {
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return {}
    }
  }
}

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------

export function resolveOpenAIAuth(
  strategy: AuthStrategy,
  auth: AuthInfo | null,
): Partial<OpenAIOptions> | undefined {
  switch (strategy.type) {
    case 'api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return undefined
    }
    case 'oauth-openai': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') {
        return {
          api_key: auth.accessToken,
          base_url: 'https://chatgpt.com/backend-api/codex',
          headers: {
            ...(auth.accountId ? { 'ChatGPT-Account-Id': auth.accountId } : {}),
          },
        }
      }
      const envKey = process.env.OPENAI_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'oauth-as-api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      return undefined
    }
    case 'oauth-anthropic': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      const envKey = process.env.ANTHROPIC_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'none':
      return {}
    case 'local-optional': {
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return {}
    }
  }
}

// ---------------------------------------------------------------------------
// OpenAI-generic
// ---------------------------------------------------------------------------

export function resolveOpenAIGenericAuth(
  strategy: AuthStrategy,
  auth: AuthInfo | null,
): Partial<OpenAIGenericOptions> | undefined {
  switch (strategy.type) {
    case 'api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return undefined
    }
    case 'oauth-as-api-key': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      return undefined
    }
    case 'oauth-openai': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') {
        return {
          api_key: auth.accessToken,
          base_url: 'https://chatgpt.com/backend-api/codex',
          headers: {
            ...(auth.accountId ? { 'ChatGPT-Account-Id': auth.accountId } : {}),
          },
        }
      }
      const envKey = process.env.OPENAI_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'oauth-anthropic': {
      if (auth?.type === 'api') return { api_key: auth.key }
      if (auth?.type === 'oauth') return { api_key: auth.accessToken }
      const envKey = process.env.ANTHROPIC_API_KEY
      if (!envKey) return undefined
      return { api_key: envKey }
    }
    case 'none':
      return {}
    case 'local-optional': {
      for (const envKey of strategy.envKeys) {
        const value = process.env[envKey]
        if (value) return { api_key: value }
      }
      return {}
    }
  }
}
