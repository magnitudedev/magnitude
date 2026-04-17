import { afterEach, describe, expect, it, vi } from 'vitest'
import { __testOnly_buildProviderOptions } from '../client-registry-builder'
import { getLowestEffortOptions } from '../reasoning-effort'
import { modelsDevCatalogSource } from '../catalog/models-dev-catalog-source'
import { openRouterCatalogSource } from '../catalog/openrouter-catalog-source'
import { staticCatalogSource } from '../catalog/static-catalog-source'
import { getProvider } from '../registry'

describe('Magnitude provider', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('builds Magnitude options from explicit API auth with text response format and stream enabled', () => {
    const options = __testOnly_buildProviderOptions(
      'magnitude',
      'glm-5.1',
      { type: 'api', key: 'test-key' },
      undefined,
      ['STOP'],
    )

    expect(options).toEqual(expect.objectContaining({
      model: 'glm-5.1',
      api_key: 'test-key',
      base_url: 'https://app.magnitude.dev/api/v1',
      max_tokens: 202000,
      stop: ['STOP'],
      stream: true,
      stream_options: { include_usage: true },
      response_format: { type: 'text' },
    }))
  })

  it('uses MAGNITUDE_API_KEY from env and supports grammar response format override', () => {
    vi.stubEnv('MAGNITUDE_API_KEY', 'env-magnitude-key')

    const options = __testOnly_buildProviderOptions(
      'magnitude',
      'qwen3.6-plus',
      null,
      undefined,
      undefined,
      'root ::= "YES" | "NO"',
    )

    expect(options).toEqual(expect.objectContaining({
      model: 'qwen3.6-plus',
      api_key: 'env-magnitude-key',
      base_url: 'https://app.magnitude.dev/api/v1',
      max_tokens: 200000,
      stream: true,
      stream_options: { include_usage: true },
      response_format: { type: 'grammar', grammar: 'root ::= "YES" | "NO"' },
    }))
  })

  it('uses Magnitude reasoning policy for minimax and non-minimax models', () => {
    expect(getLowestEffortOptions('magnitude', 'minimax-m2.7', 'openai-generic')).toEqual({
      optionsMerge: { reasoning_effort: 'low' },
      label: 'Magnitude reasoning_effort=low',
    })

    expect(getLowestEffortOptions('magnitude', 'glm-5.1', 'openai-generic')).toEqual({
      optionsMerge: { reasoning_effort: 'none' },
      label: 'Magnitude reasoning_effort=none',
    })
  })

  it('is static-only for catalog refresh behavior', () => {
    const provider = getProvider('magnitude')!
    expect(staticCatalogSource.supports(provider)).toBe(true)
    expect(modelsDevCatalogSource.supports(provider)).toBe(false)
    expect(openRouterCatalogSource.supports(provider)).toBe(false)
  })
})
