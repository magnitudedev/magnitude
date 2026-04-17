import { describe, expect, it } from 'vitest'

import { localCatalogSource } from '../catalog/local-catalog-source'
import { modelsDevCatalogSource } from '../catalog/models-dev-catalog-source'
import { openRouterCatalogSource } from '../catalog/openrouter-catalog-source'
import { staticCatalogSource } from '../catalog/static-catalog-source'
import { getProvider } from '../registry'

describe('catalog source registry contracts', () => {
  it('supports providers with expected predicates and priority ordering', () => {
    const sources = [
      staticCatalogSource,
      modelsDevCatalogSource,
      localCatalogSource,
      openRouterCatalogSource,
    ].sort((a, b) => a.priority - b.priority)

    expect(sources.map((source) => source.id)).toEqual([
      'static',
      'models.dev',
      'local-discovery',
      'openrouter-api',
    ])

    const fireworks = getProvider('fireworks-ai')!
    const magnitude = getProvider('magnitude')!
    const openrouter = getProvider('openrouter')!
    const lmstudio = getProvider('lmstudio')!
    const anthropic = getProvider('anthropic')!

    expect(staticCatalogSource.supports(anthropic)).toBe(true)
    expect(sources.find((s) => s.id === 'models.dev')!.supports(lmstudio)).toBe(false)
    expect(sources.find((s) => s.id === 'models.dev')!.supports(anthropic)).toBe(true)
    expect(sources.find((s) => s.id === 'models.dev')!.supports(fireworks)).toBe(true)
    expect(sources.find((s) => s.id === 'models.dev')!.supports(magnitude)).toBe(false)
    expect(sources.find((s) => s.id === 'local-discovery')!.supports(lmstudio)).toBe(true)
    expect(sources.find((s) => s.id === 'openrouter-api')!.supports(openrouter)).toBe(true)
    expect(sources.find((s) => s.id === 'openrouter-api')!.supports(fireworks)).toBe(false)
  })
})
