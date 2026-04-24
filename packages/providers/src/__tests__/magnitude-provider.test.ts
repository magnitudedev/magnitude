import { afterEach, describe, expect, it, vi } from 'vitest'
import { __testOnly_buildProviderOptions } from '../client-registry-builder'
import { modelsDevCatalogSource } from '../catalog/models-dev-catalog-source'
import { openRouterCatalogSource } from '../catalog/openrouter-catalog-source'
import { staticCatalogSource } from '../catalog/static-catalog-source'
import { getProvider } from '../registry'

describe('Magnitude provider', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('builds Magnitude options from explicit API auth with stream enabled', () => {
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
    }))
  })

  it('uses MAGNITUDE_API_KEY from env and supports grammar response format override', () => {
    vi.stubEnv('MAGNITUDE_API_KEY', 'env-magnitude-key')

    const options = __testOnly_buildProviderOptions(
      'magnitude',
      'glm-5.1',
      null,
      undefined,
      undefined,
      'root ::= "YES" | "NO"',
    )

    expect(options).toEqual(expect.objectContaining({
      model: 'glm-5.1',
      api_key: 'env-magnitude-key',
      base_url: 'https://app.magnitude.dev/api/v1',
      max_tokens: 202000,
      stream: true,
      stream_options: { include_usage: true },
      response_format: { type: 'grammar', grammar: 'root ::= "YES" | "NO"' },
    }))
  })

  it('applies reasoning_effort=low for minimax models', () => {
    vi.stubEnv('MAGNITUDE_API_KEY', 'test-key')

    const options = __testOnly_buildProviderOptions(
      'magnitude',
      'minimax-m2.7',
      null,
    )

    expect(options).toEqual(expect.objectContaining({
      reasoning_effort: 'low',
    }))
  })

  it('applies reasoning_effort=none for non-minimax models', () => {
    vi.stubEnv('MAGNITUDE_API_KEY', 'test-key')

    const options = __testOnly_buildProviderOptions(
      'magnitude',
      'glm-5.1',
      null,
    )

    expect(options).toEqual(expect.objectContaining({
      reasoning_effort: 'none',
    }))
  })

  it('is static-only for catalog refresh behavior', () => {
    const provider = getProvider('magnitude')!
    expect(staticCatalogSource.supports(provider)).toBe(true)
    expect(modelsDevCatalogSource.supports(provider)).toBe(false)
    expect(openRouterCatalogSource.supports(provider)).toBe(false)
  })
})
